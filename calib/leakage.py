"""M1b: 보정 세트 누출 검사 (RouterArena full 8,400 + robustness 420 대비).

1) 정규화 정확 일치 (질문, 컨텍스트 앞부분)
2) 질문 본문 임베딩 코사인 (all-MiniLM-L6-v2) ≥ 0.85 — 템플릿이 아니라 원문 Question 끼리 비교
3) source_id 중복 (보정 세트 내부)
위반 행은 --drop 시 calib_set / calib_prompts 에서 제거하고 리포트를 남긴다.

usage: uv run --native-tls python -m calib.leakage [--drop]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from calib.common import DATA_DIR, ExclusionIndex

THRESHOLD = 0.85
# 수순 문자열은 문장 임베딩 유사도가 게임 동일성을 반영하지 못한다 → 로더의 게임 단위 구조 검사로 대체
# (calib/sources/code_chess.py)
EMBEDDING_EXEMPT = {"ChessInstruct", "ChessInstruct_mcq"}
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", action="store_true")
    args = ap.parse_args()

    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    rows = [json.loads(l) for l in (DATA_DIR / "calib_set.jsonl").read_text().splitlines()]
    arena = list(load_dataset("RouteWorks/RouterArena", split="full")) + list(
        load_dataset("RouteWorks/RouterArena", split="robustness")
    )
    # 질문 + 컨텍스트 앞부분을 함께 임베딩 (고정 문구 질문만으로는 문항이 식별되지 않음)
    arena_q = [f'{r["Question"]} {str(r.get("Context") or "")[:500]}' for r in arena]
    excl = ExclusionIndex()

    exact = [r["id"] for r in rows if excl.is_excluded(r["question"])]
    ctx = [r["id"] for r in rows if r["id"] not in exact and excl.is_excluded(r["question"], r.get("context", ""))]

    model = SentenceTransformer(MODEL)
    a = model.encode(arena_q, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    c = model.encode(
        [f'{r["question"]} {str(r.get("context") or "")[:500]}' for r in rows],
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    max_sim = np.zeros(len(rows))
    for i in range(0, len(rows), 512):
        max_sim[i : i + 512] = (c[i : i + 512] @ a.T).max(axis=1)
    near = [rows[i]["id"] for i in np.where(max_sim >= THRESHOLD)[0] if rows[i]["config_name"] not in EMBEDDING_EXEMPT]

    dup_src = [k for k, n in Counter((r["source_hf"], r["source_id"]) for r in rows if r["source_id"]).items() if n > 1]

    flagged = set(exact) | set(ctx) | set(near)
    report = {
        "rows": len(rows),
        "exact_question": len(exact),
        "context_prefix": len(ctx),
        f"embedding_ge_{THRESHOLD}": len(near),
        "duplicate_source_ids": len(dup_src),
        "flagged_by_config": dict(Counter(r["config_name"] for r in rows if r["id"] in flagged)),
        "max_sim_p99": float(np.quantile(max_sim, 0.99)),
        "flagged_ids": sorted(flagged),
        "dropped": bool(args.drop),
    }
    (DATA_DIR / "leakage_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "flagged_ids"}, indent=1, ensure_ascii=False))

    if args.drop and flagged:
        for name in ("calib_set.jsonl", "calib_prompts.jsonl"):
            path = DATA_DIR / name
            kept = [l for l in path.read_text().splitlines() if json.loads(l)["id"] not in flagged]
            path.write_text("\n".join(kept) + "\n")
        print(f"dropped {len(flagged)} rows")


if __name__ == "__main__":
    main()
