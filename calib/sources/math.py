"""AIME · MATH · GSM8K · AsDiv · MathQA · FinQA 보정 세트 원천 로더.

RouterArena 원천(행 텍스트 완전일치로 확인)과 여기서 쓰는 원천:
- AIME: RouterArena = AI-MO/aimo-validation-aime(2022~2024, 90) + opencompass/AIME2025(I→II, 30) 에서 39문항.
  같은 풀의 나머지(81)를 우선 쓰고, 부족하면 di-zhang-fdu/AIME_1983_2024 의 2021년 이전 문항으로 채운다.
- MATH: RouterArena = HuggingFaceH4/MATH-500(test). 여기서는 EleutherAI/hendrycks_math train, answer = 풀이의 마지막 \\boxed{}.
- GSM8K: RouterArena = openai/gsm8k main test. 여기서는 train, answer = "####" 뒤 문자열 그대로.
- AsDiv: RouterArena = EleutherAI/asdiv validation(유일 split). Context=body, answer = answer.split()[0] (단위 제거, RouterArena와 같은 손실 규칙).
- MathQA: RouterArena = allenai/math_qa test. 여기서는 train. options 는 RouterArena 파싱 규칙(',' 분리 후 "x ) " 접두 1회 제거)을 그대로 재현.
- FinQA: RouterArena = dreamerdeo/finqa(전 split), gold_evidence 가 전부 본문(pre/post_text)인 문항만, Context = " ".join(pre_text + post_text)
  (표 제외). RouterArena 문항과 같은 문서(페이지)의 문항은 모두 제외한다.
"""
from __future__ import annotations

import random
import re
from functools import lru_cache
from typing import Callable

from calib.common import CalibRow, ExclusionIndex, normalize

OVERSAMPLE = 1.3
PARQUET = "refs/convert/parquet"  # math_qa / finqa 는 스크립트 데이터셋이라 허브의 parquet 변환본을 읽는다


@lru_cache(maxsize=None)
def _scoreable(config_name: str, answer: str) -> bool:
    """정답을 그대로 \\boxed{} 에 넣었을 때 RouterArena 채점기가 1.0 을 주는가.

    RouterArena 수학 문항(250개)은 전부 이 조건을 만족한다. 원천에는 빈 정답(FinQA)이나
    "x + y + z = 0" 처럼 math_metric 이 '=' 오른쪽만 비교해 절대 맞을 수 없는 정답(MATH)이 섞여 있어 제외한다.
    """
    from calib.scoring import score

    pred = answer.upper() if config_name == "MathQA" else answer
    return score(config_name, f"\\boxed{{{pred}}}", answer) == 1.0


def _take(pool: list[CalibRow], target: int, seed: int) -> list[CalibRow]:
    """채점 가능한 문항만, source_id 정렬 후 seed 셔플 → 앞에서 target×1.3 개."""
    pool = sorted((r for r in pool if _scoreable(r.config_name, r.answer)), key=lambda r: r.source_id)
    random.Random(seed).shuffle(pool)
    return pool[: round(target * OVERSAMPLE)]


def _ra_questions(excl: ExclusionIndex, dataset_name: str) -> set[str]:
    return {normalize(q) for q in excl.questions_by_dataset.get(dataset_name, [])}


# ---------------------------------------------------------------- AIME
AIMO_REPO = "AI-MO/aimo-validation-aime"
AIME25_REPO = "opencompass/AIME2025"
AIME_HIST_REPO = "di-zhang-fdu/AIME_1983_2024"


def _aime_id_from_url(url: str) -> str:
    # https://artofproblemsolving.com/wiki/index.php/2022_AIME_I_Problems/Problem_1 → "2022-I-1"
    m = re.search(r"(\d{4})_AIME_(I+)_Problems/Problem_(\d+)", url)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


@lru_cache(maxsize=1)
def _aime_recent() -> tuple[CalibRow, ...]:
    from datasets import load_dataset

    rows = [
        _aime_row(r["problem"], r["answer"], f"{AIMO_REPO}::train", _aime_id_from_url(r["url"]))
        for r in load_dataset(AIMO_REPO, split="train")
    ]
    for part in ("I", "II"):
        for i, r in enumerate(load_dataset(AIME25_REPO, f"AIME2025-{part}", split="test")):
            rows.append(_aime_row(r["question"], r["answer"], f"{AIME25_REPO}:AIME2025-{part}:test", f"2025-{part}-{i + 1}"))
    return tuple(rows)


@lru_cache(maxsize=1)
def _aime_hist() -> tuple[CalibRow, ...]:
    from datasets import load_dataset

    return tuple(
        _aime_row(r["Question"], r["Answer"], f"{AIME_HIST_REPO}::train", _aime_hist_id(r))
        for r in load_dataset(AIME_HIST_REPO, split="train")
    )


def _aime_hist_id(r: dict) -> str:
    part = r["Part"] if r["Part"] in ("I", "II") else "I"  # 2000년 이전은 연 1회 → "I"
    return f"{r['Year']}-{part}-{r['Problem Number']}"


def _aime_row(question: str, answer: str, source_hf: str, source_id: str) -> CalibRow:
    return CalibRow(
        dataset_name="AIME", config_name="AIME", question=question, answer=str(answer),
        source_hf=source_hf, source_id=source_id, group_id=source_id,
    )


def load_aime(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    ra_q = _ra_questions(excl, "AIME")
    recent = _aime_recent()
    ra_ids = {r.source_id for r in recent if normalize(r.question) in ra_q}
    excl.source_ids.setdefault("AIME", set()).update(ra_ids)
    avail_recent = [r for r in recent if r.source_id not in ra_ids and not excl.is_excluded(r.question)]
    # 과거 문항은 2021년 이하만 (2022~2024는 recent 풀과 같은 문항)
    avail_hist = [r for r in _aime_hist() if int(r.source_id[:4]) <= 2021 and not excl.is_excluded(r.question)]
    n = round(target * OVERSAMPLE)
    out = _take(avail_recent, target, seed)
    return out + _take(avail_hist, target, seed)[: n - len(out)]


# ---------------------------------------------------------------- MATH
MATH_REPO = "EleutherAI/hendrycks_math"
MATH_SUBJECTS = [
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]


def last_boxed(text: str) -> str | None:
    """풀이의 마지막 \\boxed{...} 내용 (중괄호 균형)."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i = start + len("\\boxed{")
    depth = 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j].strip()
    return None


@lru_cache(maxsize=1)
def _math_pool() -> tuple[CalibRow, ...]:
    from datasets import load_dataset

    rows = []
    for subj in MATH_SUBJECTS:
        for i, r in enumerate(load_dataset(MATH_REPO, subj, split="train")):
            ans = last_boxed(r["solution"])
            if not ans:
                continue
            sid = f"train/{subj}/{i}"
            rows.append(CalibRow(
                dataset_name="MATH", config_name="MATH", question=r["problem"], answer=ans,
                source_hf=f"{MATH_REPO}:{subj}:train", source_id=sid, group_id=sid,
            ))
    return tuple(rows)


def load_math(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    return _take([r for r in _math_pool() if not excl.is_excluded(r.question)], target, seed)


# ---------------------------------------------------------------- GSM8K
GSM8K_REPO = "openai/gsm8k"


@lru_cache(maxsize=1)
def _gsm8k_pool() -> tuple[CalibRow, ...]:
    from datasets import load_dataset

    return tuple(
        CalibRow(
            dataset_name="GSM8K", config_name="GSM8K", question=r["question"],
            answer=r["answer"].split("####")[-1].strip(),
            source_hf=f"{GSM8K_REPO}:main:train", source_id=f"train/{i}", group_id=f"train/{i}",
        )
        for i, r in enumerate(load_dataset(GSM8K_REPO, "main", split="train"))
    )


def load_gsm8k(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    return _take([r for r in _gsm8k_pool() if not excl.is_excluded(r.question)], target, seed)


# ---------------------------------------------------------------- AsDiv
ASDIV_REPO = "EleutherAI/asdiv"


@lru_cache(maxsize=1)
def _asdiv_pool() -> tuple[CalibRow, ...]:
    from datasets import load_dataset

    return tuple(
        CalibRow(
            dataset_name="AsDiv", config_name="AsDiv", question=r["question"], context=r["body"],
            answer=r["answer"].split()[0],  # "2 (inches)" → "2" (RouterArena 규칙)
            source_hf=f"{ASDIV_REPO}::validation", source_id=f"validation/{i}",
            group_id=f"asdiv-body/{normalize(r['body'])[:120]}",  # 같은 지문 = 같은 문제 계열
        )
        for i, r in enumerate(load_dataset(ASDIV_REPO, split="validation"))
    )


def load_asdiv(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    pool = _asdiv_pool()
    # RouterArena 문항과 지문을 공유하는 형제 문항도 제외 (짧은 지문은 is_excluded 의 context 검사에 안 걸림)
    ra_groups = {r.group_id for r in pool if excl.is_excluded(r.question, r.context)}
    ra_q = _ra_questions(excl, "AsDiv")
    ra_groups |= {r.group_id for r in pool if normalize(r.question) in ra_q}
    excl.source_ids.setdefault("AsDiv", set()).update(ra_groups)
    return _take([r for r in pool if r.group_id not in ra_groups], target, seed)


# ---------------------------------------------------------------- MathQA
MATHQA_REPO = "allenai/math_qa"


def mathqa_options(raw: str) -> list[str]:
    # RouterArena 표현 재현: "a ) 4 / 13 , b ) 1 / 13 , ..." → ["4 / 13", "1 / 13", ...]
    # (옵션 내부 쉼표·리스트 문자열 형식도 RouterArena와 똑같이 쪼개진다)
    return [re.sub(r"^[a-e] \) ", "", p.strip()) for p in raw.split(",")]


@lru_cache(maxsize=1)
def _mathqa_pool() -> tuple[CalibRow, ...]:
    from datasets import load_dataset

    return tuple(
        CalibRow(
            dataset_name="MathQA", config_name="MathQA", question=r["Problem"],
            options=mathqa_options(r["options"]), answer=r["correct"],
            source_hf=f"{MATHQA_REPO}:default:train", source_id=f"train/{i}", group_id=f"train/{i}",
        )
        for i, r in enumerate(load_dataset(MATHQA_REPO, revision=PARQUET, split="train"))
    )


def load_mathqa(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    return _take([r for r in _mathqa_pool() if not excl.is_excluded(r.question)], target, seed)


# ---------------------------------------------------------------- FinQA
FINQA_REPO = "dreamerdeo/finqa"


@lru_cache(maxsize=1)
def _finqa_pool() -> tuple[CalibRow, ...]:
    from datasets import load_dataset

    rows = []
    for split in ("train", "validation", "test"):
        for r in load_dataset(FINQA_REPO, revision=PARQUET, split=split):
            pre, post = list(r["pre_text"]), list(r["post_text"])
            text = set(pre) | set(post)
            if not all(e in text for e in r["gold_evidence"]):  # RouterArena는 본문 근거만으로 풀리는 문항만 썼다
                continue
            rows.append(CalibRow(
                dataset_name="FinQA", config_name="FinQA", question=r["question"], context=" ".join(pre + post),
                answer=r["answer"], source_hf=f"{FINQA_REPO}:default:{split}", source_id=r["id"],
                group_id=r["id"].rsplit("-", 1)[0],  # "ADI/2009/page_49.pdf-1" → 문서(페이지) 단위
            ))
    return tuple(rows)


def load_finqa(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    pool = _finqa_pool()
    ra_docs = {r.group_id for r in pool if excl.is_excluded(r.question, r.context)}
    excl.source_ids.setdefault("FinQA", set()).update(ra_docs)
    return _take([r for r in pool if r.group_id not in ra_docs], target, seed)


LOADERS: dict[str, Callable[[int, int, ExclusionIndex], list[CalibRow]]] = {
    "AIME": load_aime,
    "MATH": load_math,
    "GSM8K": load_gsm8k,
    "AsDiv": load_asdiv,
    "MathQA": load_mathqa,
    "FinQA": load_finqa,
}
