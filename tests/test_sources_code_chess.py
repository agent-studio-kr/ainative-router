"""calib/sources/code_chess.py 로더 검증 (LLM 호출 없음, RouterArena 공식 채점기로 정/오답 확인)."""
import json

import pytest

from calib.common import CalibRow, ExclusionIndex, config_for, format_prompt, normalize
from calib.scoring import score
from calib.sources import code_chess as cc

LCB_KEYS = {
    "_index", "prompt", "test", "entry_point", "canonical_solution", "task_id",
    "is_stdin", "public_test_cases", "difficulty", "global_idx",
}  # RouterArena prep_datasets.map_to_example 결과 열 (+ _index)


@pytest.fixture(scope="module")
def excl():
    return ExclusionIndex()


@pytest.fixture(scope="module")
def rows(excl):
    return {name: loader(3, 0, excl) for name, loader in cc.LOADERS.items()}


def _check_schema(r: CalibRow, excl: ExclusionIndex, name: str):
    assert r.config_name == name == config_for(r.dataset_name, bool(r.options))
    assert r.source_hf and r.source_id and r.group_id
    # Chess Context 는 KIND별 고정 지시문이라 문맥 일치 검사를 쓰면 전부 걸린다 → 문항 본문으로만 검사
    assert not excl.is_excluded(r.question, r.context if name == "LiveCodeBench" else "")
    prompt = format_prompt(r)
    assert isinstance(prompt, str) and prompt
    assert r.question[:50] in prompt.replace("{{", "{").replace("}}", "}")


@pytest.mark.parametrize("name", ["LiveCodeBench", "ChessInstruct", "ChessInstruct_mcq"])
def test_schema_and_determinism(name, rows, excl):
    got = rows[name]
    assert 1 <= len(got) <= round(3 * cc.OVERSAMPLE)
    for r in got:
        _check_schema(r, excl, name)
    again = cc.LOADERS[name](3, 0, excl)
    assert [r.source_id for r in again] == [r.source_id for r in got]
    assert len({r.group_id for r in got}) == len(got) or name == "LiveCodeBench"


# ---------------------------------------------------------------- LiveCodeBench
def test_lcb_excludes_routerarena(rows, excl):
    ra_ids = excl.source_ids["LiveCodeBench"]
    assert len(ra_ids) >= 380  # RouterArena 385문항 → release_v2 question_id (중복 문항 포함)
    ra_prefix = {normalize(q[:100]) for q in excl.questions_by_dataset["LiveCodeBench"]}
    for r in rows["LiveCodeBench"]:
        assert r.source_id not in ra_ids
        assert normalize(r.question[:100]) not in ra_prefix


def test_lcb_answer_dict_roundtrip_and_wrong_scores_zero(rows):
    for r in rows["LiveCodeBench"]:
        a = r.answer
        assert set(a) == LCB_KEYS
        assert a["is_stdin"] is r.is_stdin and a["task_id"] == r.source_id and a["prompt"] == r.question
        assert a["test"] and all({"input", "output", "testtype"} <= set(t) for t in a["test"])
        assert json.loads(json.dumps(a)) == a
        wrong = "```python\nprint(0)\n```" if r.is_stdin else "```python\ndef f(*args):\n    return 0\n```"
        assert score("LiveCodeBench", wrong, a) == 0.0
        assert score("LiveCodeBench", wrong, json.dumps(a)) == 0.0  # 직렬화된 dict 경로도 동일


# LiveCodeBench(code_generation_lite)는 정답 코드를 제공하지 않는다 (canonical_solution 없음).
# 대신 사람이 직접 작성한 정답 풀이로 채점 파이프라인(stdin/functional)이 1.0 을 내는지 확인한다.
HAND_SOLUTIONS = {
    "abc387_b": "```python\nx = int(input())\n"
    "print(sum(i * j for i in range(1, 10) for j in range(1, 10) if i * j != x))\n```",
    "3708": "```python\ndef zigzagTraversal(grid):\n    seq = []\n    for i, row in enumerate(grid):\n"
    "        seq.extend(row if i % 2 == 0 else row[::-1])\n    return seq[::2]\n```",
}


def test_lcb_hand_written_solutions_score_one():
    got = cc.lcb_rows_by_id(list(HAND_SOLUTIONS))
    assert [r.source_id for r in got] == list(HAND_SOLUTIONS)
    assert [r.is_stdin for r in got] == [True, False]
    for r in got:
        assert score("LiveCodeBench", HAND_SOLUTIONS[r.source_id], r.answer) == 1.0
        assert score("LiveCodeBench", HAND_SOLUTIONS[r.source_id].replace("i * j", "i + j").replace("[::2]", "[1::2]"), r.answer) == 0.0


# ---------------------------------------------------------------- Chess
def test_chess_representation_reproduces_routerarena():
    """변환 규칙을 RouterArena 원 문항(148개)에 적용하면 Context/Options/Answer/Metadata 가 그대로 재현돼야 한다."""
    from datasets import load_dataset

    src = {r["input"]: r for r in load_dataset(cc.CHESS_REPO, split=cc.CHESS_SPLIT)}
    ra = [r for r in load_dataset("RouteWorks/RouterArena", split="full") if "ChessInstruct" in r["Dataset name"]]
    assert len(ra) == 148
    for r in ra:
        s = src[r["Question"]]
        row = cc.chess_to_row(s["KIND"], s["task"], s["input"], s["expected_output"])
        opts = r["Options"]
        opts = json.loads(opts) if isinstance(opts, str) else list(opts or [])
        assert (row.dataset_name, row.context, row.options, row.answer) == (r["Dataset name"], r["Context"], opts, r["Answer"])
        assert row.metadata == json.loads(r["Metadata"])


@pytest.mark.parametrize("name", ["ChessInstruct", "ChessInstruct_mcq"])
def test_chess_excludes_same_game(name, rows, excl):
    ra_games = [cc._moves(q) for n, qs in excl.questions_by_dataset.items() if "ChessInstruct" in n for q in qs]
    for r in rows[name]:
        g = cc._moves(r.question)
        assert not any(cc._same_game(g, x) for x in ra_games)


def test_chess_gold_scores_one_wrong_scores_zero(rows):
    for r in rows["ChessInstruct"]:
        assert score("ChessInstruct", f"The move is \\boxed{{{r.answer}}}", r.answer) == 1.0
        wrong = "a1a1" if r.answer != "a1a1" else "h8h8"
        assert score("ChessInstruct", f"\\boxed{{{wrong}}}", r.answer) == 0.0
    for r in rows["ChessInstruct_mcq"]:
        gold = "ABC"[int(r.answer)]
        wrong = next(c for c in "ABC"[: len(r.options)] if c != gold)
        assert score("ChessInstruct_mcq", f"\\boxed{{{gold}}}", r.answer) == 1.0
        assert score("ChessInstruct_mcq", f"\\boxed{{{wrong}}}", r.answer) == 0.0
