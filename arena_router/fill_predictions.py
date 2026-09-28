"""M7: RouterArena 예측 파일의 generated_result 를 출처 보존 추론으로 채운다.

RouterArena llm_inference/run.py 는 generated_result 를 만들 때 model_used 를 버리므로(run.py:309)
자체 클라이언트(calib.infer)로 호출하고, 행마다 requested_model / model_used / provider / request_id /
invoked_at / actual_cost 를 함께 기록한다. token_usage 형식은 RouterArena _call_openrouter 와 동일.

usage: uv run --native-tls python -m arena_router.fill_predictions ainative-router [--concurrency 48] [--spend-cap 35]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os

from dotenv import load_dotenv

from calib.common import ROOT, ROUTERARENA_DIR
from calib.infer import done_ids, result_path, run_pairs

RESULTS_DIR = ROOT / "data" / "arena" / "results"


def _key(entry: dict) -> str:
    return str(entry.get("global index") or entry.get("global_index"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("router_name")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--spend-cap", type=float, default=35.0)
    ap.add_argument("--no-call", action="store_true", help="호출 없이 기존 결과만으로 채움 (끝나지 않는 호출은 실패 행으로 남김)")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    path = ROUTERARENA_DIR / "router_inference" / "predictions" / f"{args.router_name}.json"
    preds = json.loads(path.read_text())

    # 같은 (모델, 문항)은 한 번만 호출 (regular 행과 optimality 행이 겹치지 않지만 방어적으로)
    pairs, seen = [], set()
    for e in preds:
        k = (e["prediction"], _key(e))
        if k not in seen:
            seen.add(k)
            pairs.append((e["prediction"], {"id": _key(e), "prompt": e["prompt"]}))
    print(f"{len(preds)} rows, {len(pairs)} unique calls")
    if not args.no_call:
        asyncio.run(run_pairs(pairs, os.environ["OPENROUTER_API_KEY"], args.concurrency, args.spend_cap, RESULTS_DIR))

    results: dict[tuple[str, str], dict] = {}
    for model in {m for m, _ in pairs}:
        ok = done_ids(model, RESULTS_DIR)
        for rec in map(json.loads, result_path(model, RESULTS_DIR).read_text().splitlines()):
            if "error" not in rec or rec["id"] not in ok:
                results[(model, rec["id"])] = rec

    missing = 0
    for e in preds:
        rec = results.get((e["prediction"], _key(e)))
        if rec is None:
            missing += 1
            continue
        usage = rec.get("usage") or {}
        e["generated_result"] = {
            "generated_answer": rec.get("content") or "",
            "success": "error" not in rec and bool(rec.get("content")),
            "token_usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            "provider": "openrouter",
            "requested_model": rec.get("requested_model"),
            "model_used": rec.get("model_used"),
            "upstream_provider": rec.get("provider"),
            "request_id": rec.get("request_id"),
            "invoked_at": rec.get("invoked_at"),
            "actual_cost_usd": rec.get("actual_cost"),
            "error": rec.get("error"),
        }
    path.write_text(json.dumps(preds, ensure_ascii=False, indent=2))
    print(f"filled {len(preds) - missing}/{len(preds)} rows (missing {missing})")


if __name__ == "__main__":
    main()
