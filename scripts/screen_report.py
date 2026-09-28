"""MMLU-Pro 비중복 140문항 스크리닝 결과 (새 61개 + 기존 23개) — 정확도, OpenRouter 실비 $/1K."""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from calib.scoring import score  # noqa: E402

ROOT = Path(".")
gold = {json.loads(l)["id"]: json.loads(l)["answer"] for l in (ROOT / "data/probe_set.jsonl").read_text().splitlines()}
rows = []
for d in ["data/screen_results", "data/probe_results"]:
    for f in sorted((ROOT / d).glob("*.jsonl")):
        recs = {}
        for rec in map(json.loads, f.read_text().splitlines()):
            if rec["id"].startswith("mmlupro_") and "error" not in rec:
                recs[rec["id"]] = rec
        if len(recs) < 100:
            continue
        acc = sum(score("MMLUPro", r.get("content") or "", gold[i]) for i, r in recs.items()) / len(recs)
        cost = sum((r.get("actual_cost") if "actual_cost" in r else r.get("cost")) or 0 for r in recs.values()) / len(recs) * 1000
        rows.append((f.stem.replace("__", "/"), len(recs), acc, cost, d.split("/")[1]))
rows.sort(key=lambda r: -r[2])
print(f"{'model':46s} {'n':>4} {'acc':>6} {'$/1K':>7}  source")
for m, n, a, c, src in rows:
    print(f"{m:46s} {n:4d} {a*100:5.1f}% {c:7.3f}  {src}")
