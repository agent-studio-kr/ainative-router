"""artifacts/policy.json 을 원 보정 세트에 학습된 그대로 두고, 확장 세트의 새 문항(cal2_*)만으로 평가한다.

두 정책(현재 vs 이전) 모두 새 문항을 본 적이 없으므로 paired bootstrap ΔArena 가 공정한 홀드아웃 비교다.
usage: CALIB_DATA_DIR=data/calib_v2 uv run python scripts/holdout_eval.py artifacts/policy_v1.json
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from arena_router.build_policy import load_obs  # noqa: E402
from arena_router.policy import Policy, arena_score  # noqa: E402

cur = Policy.from_dict(json.load(open("artifacts/policy.json")))
prev = Policy.from_dict(json.load(open(sys.argv[1])))
obs, info = load_obs(sorted(set(cur.models) | set(prev.models)))
new = [o for o in obs if o.id.startswith("cal2_")]


def agg(rows, pol):
    w = np.array([o.weight for o in rows])
    m = [pol.route(o.task, o.domain) for o in rows]
    a = np.array([o.results[x][0] for o, x in zip(rows, m)])
    c = np.array([o.results[x][1] for o, x in zip(rows, m)])
    acc, cost = (w * a).sum() / w.sum(), (w * c).sum() / w.sum() * 1000
    return {"acc": acc, "cost_per_1k": cost, "arena": arena_score(cost, acc)}


rng = np.random.default_rng(0)
idx = np.arange(len(new))
d = []
for _ in range(2000):
    s = [new[i] for i in rng.choice(idx, len(idx))]
    d.append(agg(s, cur)["arena"] - agg(s, prev)["arena"])
report = {
    "holdout_rows": len(new),
    "load": info,
    "current": agg(new, cur),
    "previous": agg(new, prev),
    "delta_arena": agg(new, cur)["arena"] - agg(new, prev)["arena"],
    "delta_arena_ci95": list(np.percentile(d, [2.5, 97.5])),
}
print(json.dumps(report, indent=1))
open("artifacts/policy_holdout_report.json", "w").write(json.dumps(report, indent=1))
