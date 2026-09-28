"""M3 채점: data/calib/results/<model>.jsonl → data/calib/scored.jsonl

- accuracy: RouterArena 공식 scorer (calib.scoring.score). 실패/빈 응답/usage 없음 → 0점 (RouterArena 규칙과 동일)
- official_cost: RouterArena 방식 = input_tokens × in_price + output_tokens × out_price (+ reasoning gap),
  단가는 OpenRouter 가격표(data/openrouter_models.json) — 제출 시 model_cost.json 에 같은 값을 등록한다.
- actual_cost: 공급자 청구액 (참고용)

usage: uv run --native-tls python -m calib.score_results [--workers 8]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

from calib.common import DATA_DIR, ROOT, ROUTERARENA_DIR


def price_table() -> dict[str, tuple[float, float]]:
    """USD/token. 공식 채점과 같은 단가를 쓰기 위해 RouterArena model_cost.json 을 우선하고,
    거기 없는 모델(새로 등록할 모델)만 OpenRouter 가격을 쓴다 — 제출 시 같은 값을 model_cost.json 에 등록한다."""
    import sys

    sys.path.insert(0, str(ROUTERARENA_DIR))
    from universal_model_names import ModelNameManager

    models = json.loads((ROOT / "data" / "openrouter_models.json").read_text())["data"]
    table = {m["id"]: (float(m["pricing"]["prompt"]), float(m["pricing"]["completion"])) for m in models}
    ra = json.loads((ROUTERARENA_DIR / "model_cost" / "model_cost.json").read_text())
    for name in list(table):
        try:
            universal = ModelNameManager.get_universal_name(name)
        except ValueError:
            universal = name
        info = ra.get(name) or ra.get(universal)
        if info:
            table[name] = (info["input_token_price_per_million"] / 1e6, info["output_token_price_per_million"] / 1e6)
    return table


def official_cost(usage: dict, prices: tuple[float, float]) -> float:
    inp = usage.get("prompt_tokens", 0) or 0
    out = usage.get("completion_tokens", 0) or 0
    total = usage.get("total_tokens", 0) or 0
    reasoning_gap = max(0, total - inp - out)
    return inp * prices[0] + (out + reasoning_gap) * prices[1]


def _score_one(args: tuple[str, str, object]) -> float:
    from calib.scoring import score

    config, content, gold = args
    return score(config, content, gold)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rows = {r["id"]: r for r in map(json.loads, (DATA_DIR / "calib_set.jsonl").read_text().splitlines())}
    lcb_path = DATA_DIR / "lcb_answers.jsonl"
    if lcb_path.exists():
        with lcb_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["id"] in rows:
                    rows[rec["id"]]["answer"] = rec["answer"]
    prices = price_table()
    records, jobs = [], []
    for path in sorted((DATA_DIR / "results").glob("*.jsonl")):
        latest: dict[str, dict] = {}
        for rec in map(json.loads, path.read_text().splitlines()):
            if rec["id"] in rows and ("error" not in rec or rec["id"] not in latest):
                latest[rec["id"]] = rec  # 성공 기록이 있으면 성공을 우선
        for id_, rec in latest.items():
            row = rows[id_]
            usage = rec.get("usage") or {}
            valid = "error" not in rec and bool(rec.get("content")) and (usage.get("completion_tokens") or 0) > 0
            model = rec["requested_model"]
            records.append({
                "id": id_,
                "model": model,
                "config_name": row["config_name"],
                "dataset_name": row["dataset_name"],
                "group_id": row["group_id"],
                "valid": valid,
                "official_cost": official_cost(usage, prices[model]) if valid else 0.0,
                "actual_cost": rec.get("actual_cost") or 0.0,
                "finish_reason": rec.get("finish_reason"),
                "model_used": rec.get("model_used"),
            })
            jobs.append((row["config_name"], rec.get("content") or "", row["answer"]) if valid else None)

    todo = [(i, j) for i, j in enumerate(jobs) if j is not None]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        scores = list(ex.map(_score_one, [j for _, j in todo], chunksize=16))
    for rec in records:
        rec["accuracy"] = 0.0
    for (i, _), s in zip(todo, scores):
        records[i]["accuracy"] = s

    with (DATA_DIR / "scored.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"scored {len(records)} records ({len(todo)} valid)")


if __name__ == "__main__":
    main()
