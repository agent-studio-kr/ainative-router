"""v1 선택지 집합 vs v2 선택지 집합 정책을 같은 group 5-fold에서 정면 비교 (paired bootstrap ΔArena)."""
import sys

import numpy as np

sys.path.insert(0, ".")
from arena_router.build_policy import group_folds, load_obs  # noqa: E402
from arena_router.policy import arena_score, fit  # noqa: E402

V1 = "google/gemma-4-31b-it,openai/gpt-6-luna,google/gemini-3-flash-preview,deepseek/deepseek-v4.1-flash,qwen/qwen3-235b-a22b-2507".split(",")
V2 = V1 + sys.argv[1].split(",")
obs, info = load_obs(V2)  # v2 모든 선택지가 측정된 문항만
print(info)
held = {"v1": {}, "v2": {}}
for fold in group_folds(obs, 5, 0):
    test = set(fold)
    train = [o for i, o in enumerate(obs) if i not in test]
    p1, p2 = fit(train, V1, use_domain=False), fit(train, V2, use_domain=False)
    for i in fold:
        held["v1"][i] = p1.route(obs[i].task, obs[i].domain)
        held["v2"][i] = p2.route(obs[i].task, obs[i].domain)


def agg(idx, choice):
    w = np.array([obs[i].weight for i in idx])
    a = np.array([obs[i].results[choice[i]][0] for i in idx])
    c = np.array([obs[i].results[choice[i]][1] for i in idx])
    acc, cost = (w * a).sum() / w.sum(), (w * c).sum() / w.sum() * 1000
    return acc, cost, arena_score(cost, acc)


idx = np.arange(len(obs))
for k in ("v1", "v2"):
    print(k, "CV acc %.4f $/1K %.3f arena %.4f" % agg(idx, held[k]))
rng = np.random.default_rng(0)
d = [agg(b, held["v2"])[2] - agg(b, held["v1"])[2] for b in (rng.choice(idx, len(idx)) for _ in range(2000))]
lo, hi = np.percentile(d, [2.5, 97.5])
print(f"ΔArena v2−v1 = {agg(idx, held['v2'])[2] - agg(idx, held['v1'])[2]:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  → {'ADOPT v2' if lo > 0 else 'KEEP v1'}")
full = fit(obs, V2, use_domain=False)
print("v2 full-data policy:", {t[:30]: m for t, m in full.task_policy.items()})
