"""RouterArena 공식 채점기 래퍼.

third_party/RouterArena/llm_evaluation 의 scorer를 그대로 호출한다 (재구현 금지).
config_name 은 config/eval_config/zero-shot/<config_name>.json 의 파일명 (예: "MMLUPro", "Ethics_justice", "GeoGraphyData").
"""
import os
import sys
from pathlib import Path
from typing import Any

ROUTERARENA_DIR = Path(__file__).resolve().parents[1] / "third_party" / "RouterArena"
_EVAL_DIR = ROUTERARENA_DIR / "llm_evaluation"
for p in (str(_EVAL_DIR), str(ROUTERARENA_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_reasoning import get_scorers_for_dataset  # noqa: E402

import json  # noqa: E402

_CONFIG_DIR = ROUTERARENA_DIR / "config" / "eval_config" / "zero-shot"


def eval_metrics(config_name: str) -> list[str]:
    cfg = json.loads((_CONFIG_DIR / f"{config_name}.json").read_text())
    return cfg["eval_params"]["eval_metrics"]


def score(config_name: str, generated_answer: str, ground_truth: Any) -> float:
    """RouterArena와 동일하게 첫 번째 scorer로 점수를 낸다 (llm_evaluation/run.py:408)."""
    scorers = get_scorers_for_dataset(config_name, eval_metrics(config_name))
    if not scorers:
        raise ValueError(f"no scorer for {config_name}")
    scorer, _ = scorers[0]
    cwd = os.getcwd()
    try:
        # 일부 scorer가 상대 경로를 가정하므로 RouterArena 루트에서 실행
        os.chdir(ROUTERARENA_DIR)
        result = scorer(generated_answer, ground_truth)
    finally:
        os.chdir(cwd)
    return float(result[0] if isinstance(result, tuple) else result)
