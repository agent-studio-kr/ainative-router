"""보정 세트 공통 스키마 · RouterArena 동일 프롬프트 생성 · 누출 제외 인덱스.

원천 로더(calib/sources/*.py)는 모두 이 모듈의 CalibRow 를 반환한다.

    def load(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]

- target 개수의 약 1.3배까지 후보를 반환해도 된다 (선별은 build 단계).
- RouterArena에 있는 문항은 excl.is_excluded(question) 로 걸러서 반환하지 않는다.
- 원천 ID 단위 제외가 가능한 원천(문서/게임/문제 ID)은 excl.source_ids 를 함께 쓴다.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ROUTERARENA_DIR = ROOT / "third_party" / "RouterArena"
CONFIG_DIR = ROUTERARENA_DIR / "config" / "eval_config" / "zero-shot"
DATA_DIR = Path(os.environ.get("CALIB_DATA_DIR", ROOT / "data" / "calib"))  # v2 확장 실험은 data/calib_v2

sys.path.insert(0, str(ROOT / "scripts"))
import _tls  # noqa: E402,F401  사내 TLS 프록시 대응


@dataclass
class CalibRow:
    dataset_name: str  # RouterArena "Dataset name" 규칙 (예: "MMLUPro_law", "Ethics_justice", "WMT19-de-en", "QANTA_History")
    config_name: str  # eval_config 파일명 (예: "MMLUPro", "Ethics_justice", "ChessInstruct_mcq", "GeoGraphyData")
    question: str
    answer: Any  # RouterArena 채점기에 넘길 ground truth (대부분 str, LiveCodeBench는 dict)
    context: str = ""
    options: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    source_hf: str = ""  # "repo:config:split"
    source_id: str = ""  # 원천 내 고유 ID (누출 검사·재현용)
    group_id: str = ""  # CV 그룹 (같은 문서/게임/문제 계열은 같은 값)
    is_stdin: bool | None = None  # LiveCodeBench 전용

    def to_json(self) -> dict:
        return asdict(self)


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", str(text)).lower().strip()


def load_eval_params(config_name: str) -> dict:
    return json.loads((CONFIG_DIR / f"{config_name}.json").read_text())["eval_params"]


def config_manifest() -> list[str]:
    return sorted(p.stem for p in CONFIG_DIR.glob("*.json"))


def _escape_braces(text: str) -> str:
    # RouterArena prep_datasets.escape_format_braces 와 동일 동작: { → {{, } → }}, 이미 이중이면 유지
    out, i = [], 0
    while i < len(text):
        ch = text[i]
        if ch in "{}":
            if i + 1 < len(text) and text[i + 1] == ch:
                i += 2
            else:
                i += 1
            out.append(ch * 2)
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _safe_format(template: str, **kwargs: Any) -> str:
    return template.format(**{k: _escape_braces(v) if isinstance(v, str) else v for k, v in kwargs.items()})


def format_prompt(row: CalibRow) -> str:
    """third_party/RouterArena/scripts/process_datasets/prep_datasets.py 의 프롬프트 생성 규칙을 그대로 따른다."""
    params = load_eval_params(row.config_name)
    options_str = "".join(
        f"{'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[i] if i < 26 else '-'}. {opt}\n" for i, opt in enumerate(row.options)
    )
    context_for_prompt = row.context if row.context != "" else "None"
    name = row.config_name
    if name == "LiveCodeBench":
        template = params.get("is_stdin_prompt") if row.is_stdin else params.get("not_is_stdin_prompt")
        prompt = _safe_format(template or "{Question}", Question=row.question)
    elif name == "SuperGLUE-RC":
        prompt = _safe_format(params.get("prompt", "{Question}"), Question=row.question, Answer="")
    elif name == "SuperGLUE-Wic":
        prompt = _safe_format(params.get("prompt", "{Question}"), Question=row.question, Context=row.context)
    elif not row.options:
        prompt = _safe_format(params.get("prompt", "{Question}"), Context=context_for_prompt, Question=row.question)
    else:
        prompt = _safe_format(
            params.get("prompt", "{Question}"), Context=context_for_prompt, Question=row.question, Options=options_str
        )
    if len(prompt) > 10000:
        prompt = f"{prompt[:5000]}...{prompt[-5000:]}"
    return prompt


TEMPLATE_CONTEXT_MIN_REPEATS = 3


class ExclusionIndex:
    """RouterArena full(8,400) + robustness(420) 문항을 제외 목록으로만 보관한다 (학습/튜닝 용도 사용 금지)."""

    def __init__(self) -> None:
        from datasets import load_dataset

        full = load_dataset("RouteWorks/RouterArena", split="full")
        robust = load_dataset("RouteWorks/RouterArena", split="robustness")
        self.questions_by_dataset: dict[str, list[str]] = {}
        q_counts: dict[str, int] = {}
        ctx_counts: dict[str, int] = {}
        for r in full:
            q = normalize(r["Question"])
            q_counts[q] = q_counts.get(q, 0) + 1
            if r["Context"]:
                key = normalize(r["Context"])[:300]
                ctx_counts[key] = ctx_counts.get(key, 0) + 1
            self.questions_by_dataset.setdefault(r["Dataset name"], []).append(r["Question"])
        # 3회 이상 반복되는 컨텍스트는 원천 공통 지시문(템플릿)이므로 누출 판정에서 뺀다 (예: ChessInstruct).
        # 컨텍스트 비교는 정규화 후 앞 300자, 50자 초과일 때만 — 짧은 전제문 원천은 로더가 따로 제외한다.
        self._contexts: set[str] = {k for k, n in ctx_counts.items() if n < TEMPLATE_CONTEXT_MIN_REPEATS}
        for r in robust:
            q = normalize(r["Question"])
            q_counts[q] = q_counts.get(q, 0) + 1
        # 3회 이상 반복되는 질문은 고정 문구(예: COPA "What was the cause?")라 문항 식별자가 아니다.
        # 이런 원천은 로더가 전제문/지문 단위로 제외한다.
        self._questions: set[str] = {q for q, n in q_counts.items() if n < TEMPLATE_CONTEXT_MIN_REPEATS}
        # 원천 로더가 채우는 원천 ID 단위 제외 목록 (예: NarrativeQA 문서 ID, LCB question_id)
        self.source_ids: dict[str, set[str]] = {}

    def is_excluded(self, question: str, context: str = "") -> bool:
        if normalize(question) in self._questions:
            return True
        if context and normalize(context)[:300] in self._contexts and len(normalize(context)) > 50:
            return True
        return False


def config_for(dataset_name: str, has_options: bool) -> str:
    """RouterArena Dataset name → eval_config 이름 (prep_datasets.py 규칙과 동일)."""
    if "Ethics" in dataset_name:
        return dataset_name
    if "ChessInstruct" in dataset_name:
        return "ChessInstruct_mcq" if has_options else "ChessInstruct"
    base = dataset_name.split("_", 1)[0]
    return "GeoGraphyData" if base == "GeoGraphyData" else base
