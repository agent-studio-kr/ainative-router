"""v2 채택 전 추가 점검: (1) 동결 v1 정책 vs v2 CV, (2) 도메인 변형 CV, (3) 과제군별 모델 추정치."""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from arena_router.build_policy import group_folds, load_obs  # noqa: E402
from arena_router.policy import Policy, arena_score, fit  # noqa: E402

V1 = "google/gemma-4-31b-it,openai/gpt-6-luna,google/gemini-3-flash-preview,deepseek/deepseek-v4.1-flash,qwen/qwen3-235b-a22b-2507".split(",")
V2 = V1 + ["google/gemma-4-31b-it@high", "deepseek/deepseek-v4-flash-0731"]
obs, info = load_obs(V2)
frozen = Policy.from_dict(json.load(open("artifacts/policy_v1.json")))
held = {"frozen_v1": {}, "v2": {}, "v2_domain": {}}
for fold in group_folds(obs, 5, 0):
    test = set(fold)
    train = [o for i, o in enumerate(obs) if i not in test]
    p2, p2d = fit(train, V2, use_domain=False), fit(train, V2, use_domain=True)
    for i in fold:
        held["frozen_v1"][i] = frozen.route(obs[i].task, obs[i].domain)
        held["v2"][i] = p2.route(obs[i].task, obs[i].domain)
        held["v2_domain"][i] = p2d.route(obs[i].task, obs[i].domain)


def agg(idx, choice):
    w = np.array([obs[i].weight for i in idx])
    a = np.array([obs[i].results[choice[i]][0] for i in idx])
    c = np.array([obs[i].results[choice[i]][1] for i in idx])
    acc, cost = (w * a).sum() / w.sum(), (w * c).sum() / w.sum() * 1000
    return acc, cost, arena_score(cost, acc)


def delta(idx, a, b, label):
    rng = np.random.default_rng(0)
    d = [agg(s, held[a])[2] - agg(s, held[b])[2] for s in (rng.choice(idx, len(idx)) for _ in range(2000))]
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"{label}: Δ {agg(idx, held[a])[2] - agg(idx, held[b])[2]:+.4f} [{lo:+.4f}, {hi:+.4f}]")


all_idx = np.arange(len(obs))
new_idx = np.array([i for i, o in enumerate(obs) if o.id.startswith("cal2_")])
for k in held:
    print(k, "all  acc %.4f $/1K %.3f arena %.4f" % agg(all_idx, held[k]))
    print(k, "new  acc %.4f $/1K %.3f arena %.4f" % agg(new_idx, held[k]))
delta(all_idx, "v2", "frozen_v1", "v2 − frozen v1 (all rows; v1 in-sample on old rows)")
delta(new_idx, "v2", "frozen_v1", "v2 − frozen v1 (new rows only; both out-of-sample for v1)")
delta(all_idx, "v2_domain", "v2", "v2×domain − v2")

# 과제군별 모델 추정치 (가중 정확도, $/1K) — 선택 근거 확인
from collections import defaultdict  # noqa: E402
by = defaultdict(list)
for o in obs:
    by[o.task].append(o)
for t in sorted(by, key=lambda t: -sum(o.weight for o in by[t]))[:6]:
    os_ = by[t]
    w = np.array([o.weight for o in os_])
    print(f"\n{t[:50]} (n={len(os_)}, weight share {w.sum() / sum(o.weight for o in obs):.1%})")
    for m in V2:
        a = np.array([o.results[m][0] for o in os_]); c = np.array([o.results[m][1] for o in os_])
        print(f"  {m:40s} acc {(w * a).sum() / w.sum():.3f}  $/1K {(w * c).sum() / w.sum() * 1000:.3f}")
