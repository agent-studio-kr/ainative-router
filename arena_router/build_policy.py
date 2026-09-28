"""M5: 보정 결과로 정책을 만들고 group k-fold로 검증한다.

입력: data/calib/scored.jsonl, calib_set.jsonl, calib_prompts.jsonl, calib/targets.json
출력: artifacts/policy.json, artifacts/policy_report.json

검증: group k-fold(같은 문서/게임/문제 계열은 같은 fold) — fold마다 train으로 정책·최고 단일 모델 선택,
held-out 문항에 적용. 모든 fold의 held-out을 모아 paired bootstrap으로 ΔArena 95% CI.

usage: uv run --native-tls python -m arena_router.build_policy [--cv 5] [--no-domain]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict

import numpy as np

from arena_router.policy import Obs, Policy, arena_score, best_single, evaluate, fit
from arena_router.signals import SignalExtractor
from calib.common import DATA_DIR, ROOT

ARTIFACTS = ROOT / "artifacts"
SIGNALS_CACHE = DATA_DIR / "signals.jsonl"


def load_signals(prompts: list[dict]) -> dict[str, dict]:
    if SIGNALS_CACHE.exists():
        cached = {r["id"]: r for r in map(json.loads, SIGNALS_CACHE.read_text().splitlines())}
        if all(p["id"] in cached for p in prompts):
            return cached
    ex = SignalExtractor([], [])  # 보정 세트는 전부 템플릿 매칭 (kNN 불필요)
    sigs = ex.extract([p["prompt"] for p in prompts])
    out = {}
    with SIGNALS_CACHE.open("w", encoding="utf-8") as f:
        for p, s in zip(prompts, sigs):
            rec = {"id": p["id"], "task": s.task, "task_source": s.task_source, "domain": s.domain, "domain_conf": s.domain_conf}
            out[p["id"]] = rec
            f.write(json.dumps(rec) + "\n")
    return out


def load_obs(models: list[str]) -> tuple[list[Obs], dict]:
    rows = {r["id"]: r for r in map(json.loads, (DATA_DIR / "calib_set.jsonl").read_text().splitlines())}
    prompts = [json.loads(l) for l in (DATA_DIR / "calib_prompts.jsonl").read_text().splitlines()]
    sigs = load_signals(prompts)
    targets = json.loads((ROOT / "calib" / "targets.json").read_text())
    ra_counts = targets["routerarena_counts"]
    calib_counts = Counter(r["config_name"] for r in rows.values())
    weight = {c: ra_counts.get(c, 0) / n for c, n in calib_counts.items()}

    res: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for rec in map(json.loads, (DATA_DIR / "scored.jsonl").read_text().splitlines()):
        res[rec["id"]][rec["model"]] = (rec["accuracy"], rec["official_cost"])

    obs, missing = [], 0
    for id_, r in rows.items():
        if not all(m in res[id_] for m in models):
            missing += 1
            continue
        if weight[r["config_name"]] == 0:
            continue
        obs.append(Obs(id_, sigs[id_]["task"], sigs[id_]["domain"], weight[r["config_name"]], r["group_id"] or id_, res[id_]))
    return obs, {"rows": len(rows), "missing_model_results": missing, "used": len(obs)}


def group_folds(obs: list[Obs], k: int, seed: int) -> list[list[int]]:
    groups = sorted({o.group_id for o in obs})
    random.Random(seed).shuffle(groups)
    fold_of = {g: i % k for i, g in enumerate(groups)}
    folds: list[list[int]] = [[] for _ in range(k)]
    for i, o in enumerate(obs):
        folds[fold_of[o.group_id]].append(i)
    return folds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="google/gemma-4-31b-it,openai/gpt-6-luna,google/gemini-3-flash-preview,deepseek/deepseek-v4.1-flash,qwen/qwen3-235b-a22b-2507")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--no-domain", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    models = args.models.split(",")
    use_domain = not args.no_domain

    obs, load_info = load_obs(models)
    print(json.dumps(load_info))

    # 1) group k-fold: held-out 문항별 (router 선택, 단일 모델 선택)
    held_router: dict[int, str] = {}
    held_single: dict[int, str] = {}
    for fold in group_folds(obs, args.cv, args.seed):
        test = set(fold)
        train = [o for i, o in enumerate(obs) if i not in test]
        p = fit(train, models, use_domain)
        single = best_single(train, models)
        for i in fold:
            held_router[i] = p.route(obs[i].task, obs[i].domain)
            held_single[i] = single

    def agg(idx: np.ndarray, choice: dict[int, str]) -> tuple[float, float, float]:
        w = np.array([obs[i].weight for i in idx])
        a = np.array([obs[i].results[choice[i]][0] for i in idx])
        c = np.array([obs[i].results[choice[i]][1] for i in idx])
        acc, cost = float((w * a).sum() / w.sum()), float((w * c).sum() / w.sum() * 1000)
        return acc, cost, arena_score(cost, acc)

    all_idx = np.arange(len(obs))
    r_acc, r_cost, r_arena = agg(all_idx, held_router)
    s_acc, s_cost, s_arena = agg(all_idx, held_single)
    rng = np.random.default_rng(args.seed)
    deltas = []
    for _ in range(args.bootstrap):
        b = rng.choice(all_idx, size=len(all_idx), replace=True)
        deltas.append(agg(b, held_router)[2] - agg(b, held_single)[2])
    lo, hi = np.percentile(deltas, [2.5, 97.5])

    # 2) 전체 데이터로 최종 정책
    final = fit(obs, models, use_domain)
    f_acc, f_cost, f_arena = evaluate(final, obs)
    single_all = {m: evaluate(Policy(models, m), obs) for m in models}

    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "policy.json").write_text(json.dumps(final.to_dict(), indent=1))
    usage = Counter(final.route(o.task, o.domain) for o in obs)
    report = {
        "load": load_info,
        "use_domain": use_domain,
        "cv": {
            "router": {"acc": r_acc, "cost_per_1k": r_cost, "arena": r_arena},
            "best_single_per_fold": {"acc": s_acc, "cost_per_1k": s_cost, "arena": s_arena, "models": dict(Counter(held_single.values()))},
            "delta_arena": r_arena - s_arena,
            "delta_arena_ci95": [float(lo), float(hi)],
        },
        "final_in_sample": {"acc": f_acc, "cost_per_1k": f_cost, "arena": f_arena, "lambda": final.lam},
        "single_models_in_sample": {m: {"acc": a, "cost_per_1k": c, "arena": s} for m, (a, c, s) in single_all.items()},
        "final_route_share_unweighted": dict(usage),
    }
    (ARTIFACTS / "policy_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["cv"], indent=1))
    print(json.dumps(report["final_in_sample"], indent=1))
    for m, v in sorted(single_all.items(), key=lambda kv: -kv[1][2]):
        print(f"  single {m:34s} acc {v[0]*100:5.1f}  $/1K {v[1]:.3f}  arena {v[2]*100:5.2f}")


if __name__ == "__main__":
    main()
