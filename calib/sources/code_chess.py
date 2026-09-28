"""LiveCodeBench · ChessInstruct · ChessInstruct_mcq 보정 세트 원천 로더.

- LiveCodeBench: RouterArena는 lighteval/code_generation_lite release_v2(콘테스트 ≤2024-05-25)에서 385문항을 썼다.
  여기서는 v6 증분(release_v6에서 새로 추가된 175문항, 2025-01~04)만 쓴다.
  answer 는 RouterArena prep_datasets.map_to_example 이 만드는 ./dataset/livecodebench 행과 같은 dict
  (test = 디코딩된 private_test_cases) 이므로 code_accuracy 에 그대로 넘긴다.
- ChessInstruct(_mcq): RouterArena는 Thytu/ChessInstruct test split 에서 4개 KIND만 뽑아 변환했다.
  같은 split 에서 RouterArena 문항과 같은 게임(수순이 서로 접두사 관계)을 모두 제외하고 동일 표현으로 재구성한다.
"""
from __future__ import annotations

import base64
import json
import pickle
import random
import zlib
from dataclasses import replace
from functools import lru_cache
from typing import Callable

from calib.common import CalibRow, ExclusionIndex, normalize

OVERSAMPLE = 1.3

# ---------------------------------------------------------------- LiveCodeBench
LCB_REPO = "lighteval/code_generation_lite"
LCB_FILE = "v6/test-00000-of-00001.parquet"  # release_v6 에서 새로 추가된 문항만 (release_v5 이후분)
LCB_RA_RELEASE = "release_v2"  # RouterArena prep_datasets.py 가 쓴 릴리스 (제외 ID 산출용, 메타 열만 읽음)
LCB_META_COLS = [
    "question_title", "question_content", "platform", "question_id", "contest_id",
    "contest_date", "starter_code", "difficulty", "public_test_cases", "metadata",
]


def _decode_private(encoded: str) -> list[dict]:
    # prep_datasets.translate_private_test_cases 와 동일
    return json.loads(pickle.loads(zlib.decompress(base64.b64decode(encoded))))


def _has_test_type(tests: str, kind: str) -> bool:
    # prep_datasets.has_test_type 와 동일
    try:
        return any(t.get("testtype") == kind for t in json.loads(tests))
    except (json.JSONDecodeError, TypeError, AttributeError):
        return False


@lru_cache(maxsize=1)
def _lcb_path() -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(LCB_REPO, LCB_FILE, repo_type="dataset")


@lru_cache(maxsize=1)
def _lcb_meta() -> list[dict]:
    """private_test_cases(수백 MB) 를 빼고 메타 열만 읽는다."""
    import pyarrow.parquet as pq

    rows = pq.read_table(_lcb_path(), columns=LCB_META_COLS).to_pylist()
    for i, r in enumerate(rows):
        r["_row"] = i
    return rows


@lru_cache(maxsize=1)
def _routerarena_lcb_ids(ra_questions: tuple[str, ...]) -> tuple[frozenset, frozenset, str]:
    """RouterArena LCB 문항 → release_v2 의 question_id / title (본문 완전일치 또는 앞 100자 매칭, prep_datasets 규칙)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    rows: list[dict] = []
    for f in sorted(fs.glob(f"datasets/{LCB_REPO}/{LCB_RA_RELEASE}/*.parquet")):
        with fs.open(f, "rb") as fh:
            rows += pq.read_table(fh, columns=["question_id", "question_title", "question_content", "contest_date"]).to_pylist()
    ids, titles = set(), set()
    for q in ra_questions:
        for r in rows:
            c = r["question_content"]
            if c == q or c.startswith(q[:100]) or q[:100] in c:
                ids.add(r["question_id"])
                titles.add(normalize(r["question_title"]))
                break
    max_date = max(r["contest_date"] for r in rows)
    return frozenset(ids), frozenset(titles), max_date


def lcb_row(meta: dict, test: list[dict]) -> CalibRow:
    is_stdin = _has_test_type(meta["public_test_cases"], "stdin")
    answer = {  # prep_datasets.map_to_example 과 같은 키
        "_index": str(meta["_row"]),
        "prompt": meta["question_content"],
        "test": test,
        "entry_point": meta["starter_code"],
        "canonical_solution": "",
        "task_id": meta["question_id"],
        "is_stdin": is_stdin,
        "public_test_cases": meta["public_test_cases"],
        "difficulty": meta["difficulty"],
        "global_idx": None,  # RouterArena 전용 인덱스 — 보정 세트에는 없음
    }
    md = json.loads(meta["metadata"] or "{}")
    md.update(platform=meta["platform"], contest_date=meta["contest_date"], question_title=meta["question_title"])
    return CalibRow(
        dataset_name="LiveCodeBench",
        config_name="LiveCodeBench",
        question=meta["question_content"],
        answer=answer,
        context=meta["starter_code"],  # RouterArena Context = starter_code
        metadata=md,
        source_hf=f"{LCB_REPO}:{LCB_FILE.split('/')[0]}:test",
        source_id=meta["question_id"],
        group_id=f"lcb:{meta['contest_id']}",
        is_stdin=is_stdin,
    )


def lcb_rows_by_id(question_ids: list[str]) -> list[CalibRow]:
    """지정한 question_id 들만 private test 를 디코딩해 CalibRow 로 만든다 (메모리 절약)."""
    import pyarrow.parquet as pq

    wanted = set(question_ids)
    metas = [m for m in _lcb_meta() if m["question_id"] in wanted]
    metas.sort(key=lambda m: question_ids.index(m["question_id"]))
    col = pq.read_table(_lcb_path(), columns=["private_test_cases"])["private_test_cases"]
    return [lcb_row(m, _decode_private(col[m["_row"]].as_py())) for m in metas]


def lcb_available(excl: ExclusionIndex) -> list[dict]:
    ra_q = tuple(excl.questions_by_dataset.get("LiveCodeBench", []))
    ra_ids, ra_titles, ra_max_date = _routerarena_lcb_ids(ra_q)
    excl.source_ids.setdefault("LiveCodeBench", set()).update(ra_ids)
    ra_prefix = {normalize(q[:100]) for q in ra_q}
    out = []
    for m in _lcb_meta():
        if (
            m["question_id"] in excl.source_ids["LiveCodeBench"]
            or normalize(m["question_title"]) in ra_titles
            or m["contest_date"] <= ra_max_date
            or excl.is_excluded(m["question_content"])
            or normalize(m["question_content"][:100]) in ra_prefix
        ):
            continue
        out.append(m)
    return out


def load_livecodebench(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    pool = sorted(lcb_available(excl), key=lambda m: m["question_id"])
    random.Random(seed).shuffle(pool)
    picked = [m["question_id"] for m in pool[: round(target * OVERSAMPLE)]]
    return lcb_rows_by_id(picked)


# ---------------------------------------------------------------- ChessInstruct
CHESS_REPO = "Thytu/ChessInstruct"
CHESS_SPLIT = "test"  # RouterArena 148문항 전부 test split 에서 완전일치로 확인됨
# RouterArena 구성 비율 (full split): ChessInstruct = next_best 38 / last_move 30, _mcq = final_score 40 / advantaged 40
FREE_KINDS = {"FIND_NEXT_BEST_MOVE": 38, "FIND_LAST_MOVE": 30}
MCQ_KINDS = {"FIND_FINAL_SCORE": 40, "FIND_ADVANTAGED_PLAYER": 40}
SCORE_OPTIONS = ["White Wins", "Black Wins", "Draw"]
SCORE_TO_IDX = {"1-0": "0", "0-1": "1", "1/2-1/2": "2"}
SIDE_OPTIONS = ["White", "Black"]


def _moves(question: str) -> tuple[str, ...]:
    return tuple(m for m in json.loads(question)["moves"] if m != "?")


def _same_game(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """한 게임에서 파생된 문항은 수순이 서로 접두사 관계다 (짧은 공통 오프닝도 보수적으로 같은 게임 취급)."""
    n = min(len(a), len(b))
    return n > 0 and a[:n] == b[:n]


def chess_to_row(kind: str, task: str, question: str, expected: str) -> CalibRow:
    """Thytu/ChessInstruct 행 → RouterArena 표현 (full split 148문항과 대조해 확인한 규칙)."""
    out = json.loads(expected)
    options: list[str] = []
    context = task
    if kind == "FIND_NEXT_BEST_MOVE":
        answer = out["next best move"]
    elif kind == "FIND_LAST_MOVE":
        answer = out["missing move"]
    elif kind == "FIND_FINAL_SCORE":
        context = task.split("\nOutput Format:")[0]  # RouterArena는 MCQ 변환 시 Output Format 줄을 제거
        options, answer = SCORE_OPTIONS, SCORE_TO_IDX[out["score"]]
    elif kind == "FIND_ADVANTAGED_PLAYER":
        options, answer = SIDE_OPTIONS, str(SIDE_OPTIONS.index(out["Most advantaged"]))
    else:
        raise ValueError(kind)
    return CalibRow(
        dataset_name="ChessInstruct_mcq" if options else "ChessInstruct",
        config_name="ChessInstruct_mcq" if options else "ChessInstruct",
        question=question,
        answer=answer,
        context=context,
        options=list(options),
        metadata={"puzzle_kind": kind.lower()},
        source_hf=f"{CHESS_REPO}:default:{CHESS_SPLIT}",
    )


@lru_cache(maxsize=1)
def _chess_pool(ra_questions: tuple[str, ...]) -> tuple[CalibRow, ...]:
    """RouterArena 문항·같은 게임 제외 후 후보 전체. group_id 는 접두사 관계 union-find 로 묶은 게임 ID."""
    from datasets import load_dataset

    ds = load_dataset(CHESS_REPO, split=CHESS_SPLIT)
    ra_games = [_moves(q) for q in ra_questions]
    ra_set = set(ra_questions)
    rows: list[CalibRow] = []
    games: list[tuple[str, ...]] = []
    for i, r in enumerate(ds):
        if r["KIND"] not in FREE_KINDS and r["KIND"] not in MCQ_KINDS:
            continue
        g = _moves(r["input"])
        if not g or r["input"] in ra_set or any(_same_game(g, x) for x in ra_games):
            continue
        row = chess_to_row(r["KIND"], r["task"], r["input"], r["expected_output"])
        row.source_id = f"{CHESS_SPLIT}:{i}"
        rows.append(row)
        games.append(g)
    parent = list(range(len(rows)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if _same_game(games[i], games[j]):
                parent[max(find(i), find(j))] = min(find(i), find(j))
    for i, row in enumerate(rows):
        row.group_id = f"chess:{rows[find(i)].source_id}"
    return tuple(rows)


def chess_available(excl: ExclusionIndex, config_name: str) -> list[CalibRow]:
    ra_q = tuple(
        q for name, qs in sorted(excl.questions_by_dataset.items()) if "ChessInstruct" in name for q in qs
    )
    return [r for r in _chess_pool(ra_q) if r.config_name == config_name and not excl.is_excluded(r.question)]


def _load_chess(config_name: str, kinds: dict[str, int]) -> Callable[[int, int, ExclusionIndex], list[CalibRow]]:
    def load(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
        n = round(target * OVERSAMPLE)
        pool = chess_available(excl, config_name)
        rng = random.Random(seed)
        total = sum(kinds.values())
        out: list[CalibRow] = []
        used_groups: set[str] = set()
        for k, (kind, w) in enumerate(kinds.items()):
            quota = n - len(out) if k == len(kinds) - 1 else round(n * w / total)
            cands = [r for r in pool if r.metadata["puzzle_kind"] == kind.lower()]
            rng.shuffle(cands)
            for r in cands:  # 한 게임에서 한 문항만
                if quota == 0:
                    break
                if r.group_id not in used_groups:
                    used_groups.add(r.group_id)
                    out.append(replace(r, options=list(r.options), metadata=dict(r.metadata)))
                    quota -= 1
        return out

    return load


LOADERS: dict[str, Callable[[int, int, ExclusionIndex], list[CalibRow]]] = {
    "LiveCodeBench": load_livecodebench,
    "ChessInstruct": _load_chess("ChessInstruct", FREE_KINDS),
    "ChessInstruct_mcq": _load_chess("ChessInstruct_mcq", MCQ_KINDS),
}
