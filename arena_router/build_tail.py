"""추론 길이 예측기(arena_router.tail)를 외부 보정 문항으로 학습하고 정책에 tail_rule 을 추가한다.

- 학습 문항: 보정셋에서 --task 범주(원천 기준)이면서 v4.1-flash 세 실행(low/기본/cap1000) 추론 토큰이 모두 있는 문항
- 목표: 세 실행 평균 추론 토큰의 log1p
- 임계값: group 5-fold out-of-fold 예측의 상위 --top 분위수 (보정셋에서 약 --top 비율이 전환되도록)

usage: CALIB_DATA_DIR=data/calib_v3 uv run --native-tls python -m arena_router.build_tail \
         --task mcq_knowledge --model deepseek/deepseek-v4-flash-0731@low --top 0.10 --out artifacts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from arena_router.build_policy import ARTIFACTS, group_folds, load_embeddings
from arena_router.signals import CATEGORY_OF_SOURCE
from arena_router.tail import TailPredictor
from calib.common import DATA_DIR

RUNS = ["deepseek/deepseek-v4.1-flash@low", "deepseek/deepseek-v4.1-flash", "deepseek/deepseek-v4.1-flash@cap1000"]


def reasoning_tokens(model: str) -> dict[str, int]:
    """보정 추론에서 provider 가 보고한 추론 토큰 수 (results/*.jsonl 의 usage 에서 뽑은 reasoning_tokens.jsonl)."""
    rows = map(json.loads, (DATA_DIR / "reasoning_tokens.jsonl").read_text().splitlines())
    return {r["id"]: r["reasoning_tokens"] for r in rows if r["model"] == model}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mcq_knowledge")
    ap.add_argument("--model", required=True, help="예측 꼬리 문항에 보낼 선택지")
    ap.add_argument("--top", type=float, default=0.10)
    ap.add_argument("--out", default=str(ARTIFACTS))
    args = ap.parse_args()
    out = Path(args.out)

    rows = {r["id"]: r for r in map(json.loads, (DATA_DIR / "calib_set.jsonl").read_text().splitlines())}
    prompts = [json.loads(l) for l in (DATA_DIR / "calib_prompts.jsonl").read_text().splitlines()]
    emb = load_embeddings(prompts)
    text = {p["id"]: p["prompt"] for p in prompts}
    rt = [reasoning_tokens(m) for m in RUNS]
    ids = [i for i in rows if CATEGORY_OF_SOURCE[rows[i]["config_name"]] == args.task and all(i in d for d in rt)]
    target = np.log1p(np.mean([[d[i] for d in rt] for i in ids], axis=1))
    P, X = [text[i] for i in ids], np.array([emb[i] for i in ids])

    class _G:
        def __init__(self, g: str) -> None:
            self.group_id = g

    oof = np.zeros(len(ids))
    for fold in group_folds([_G(rows[i]["group_id"] or i) for i in ids], 5, 0):
        tr = np.setdiff1d(np.arange(len(ids)), fold)
        oof[fold] = TailPredictor.fit([P[j] for j in tr], X[tr], target[tr]).predict([P[j] for j in fold], X[fold])
    threshold = float(np.quantile(oof, 1 - args.top))

    from sklearn.metrics import roc_auc_score

    tail = np.array([rt[0][i] > 2000 for i in ids])
    print(f"n={len(ids)}  OOF AUROC(low 실행 추론 >2000) {roc_auc_score(tail, oof):.3f}  threshold {threshold:.3f}")

    TailPredictor.fit(P, X, target).save(out / "tail_model.npz")
    policy = json.loads((out / "policy.json").read_text())
    if args.model not in policy["models"]:
        policy["models"].append(args.model)
    policy["tail_rule"] = {"task": args.task, "model": args.model, "threshold": threshold, "top_fraction_on_calibration": args.top}
    (out / "policy.json").write_text(json.dumps(policy, indent=1))
    print(json.dumps(policy["tail_rule"]))


if __name__ == "__main__":
    main()
