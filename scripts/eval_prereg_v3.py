"""docs/prereg_v3.md 의 판정 규칙을 그대로 실행한다 (후보 B·C vs 동결 v2 정책 A, cal3_* 행만 평가).

usage: CALIB_DATA_DIR=data/calib_v3 uv run python scripts/eval_prereg_v3.py
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from arena_router.build_policy import load_obs  # noqa: E402
from arena_router.policy import Policy, arena_score, fit  # noqa: E402

A = Policy.from_dict(json.load(open("artifacts/policy.json")))
MODELS = list(A.models)
obs, info = load_obs(MODELS)
train = [o for o in obs if not o.id.startswith("cal3_")]
test = [o for o in obs if o.id.startswith("cal3_")]
cands = {"B": fit(train, MODELS, use_domain=False), "C": fit(train, MODELS, use_domain=True)}


def agg(rows, pol):
    w = np.array([o.weight for o in rows])
    m = [pol.route(o.task, o.domain) for o in rows]
    a = np.array([o.results[x][0] for o, x in zip(rows, m)])
    c = np.array([o.results[x][1] for o, x in zip(rows, m)])
    acc, cost = (w * a).sum() / w.sum(), (w * c).sum() / w.sum() * 1000
    return {"acc": acc, "cost_per_1k": cost, "arena": arena_score(cost, acc)}


rng = np.random.default_rng(0)
samples = [rng.choice(len(test), len(test)) for _ in range(2000)]
report = {"load": info, "train_rows": len(train), "test_rows": len(test), "A": agg(test, A)}
passed = {}
for k, p in cands.items():
    d = [agg([test[i] for i in s], p)["arena"] - agg([test[i] for i in s], A)["arena"] for s in samples]
    delta = agg(test, p)["arena"] - report["A"]["arena"]
    lo, hi = np.percentile(d, [1.25, 98.75])
    report[k] = {**agg(test, p), "delta_arena": delta, "ci97_5": [lo, hi], "pass": bool(lo > 0),
                 "changed_cells": {t: (A.task_policy.get(t, A.default_model), m) for t, m in p.task_policy.items() if A.task_policy.get(t, A.default_model) != m}}
    if lo > 0:
        passed[k] = delta
report["decision"] = max(passed, key=passed.get) if passed else "A (keep v2)"
print(json.dumps(report, indent=1, default=str))
open("artifacts/prereg_v3_report.json", "w").write(json.dumps(report, indent=1, default=str))
if passed:
    json.dump(cands[report["decision"]].to_dict(), open(f"artifacts/policy_v3_{report['decision']}.json", "w"), indent=1)
