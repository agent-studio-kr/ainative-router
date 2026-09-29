"""보정 결과로 내용 범주 분류기와 정책을 만들고 group k-fold로 검증한다.

입력: $CALIB_DATA_DIR/{scored,calib_set,calib_prompts}.jsonl (외부 보정 문항만)
출력: artifacts/policy.json, artifacts/category_clf.npz(+.classes.json), artifacts/policy_report.json

- 셀 = 질문 본문에서 예측한 내용 범주. 정책 학습 문항에는 out-of-fold 예측 범주를 넣어 분류기 오류를 반영한다.
- 가중치: 외부 원천마다 합이 1 (원천 균등). 벤치마크의 원천 비율은 쓰지 않는다.
- 검증: group k-fold — fold마다 분류기·정책·최고 단일 모델을 train으로만 다시 맞추고 held-out에 적용,
  paired bootstrap ΔArena 95% CI.

usage: CALIB_DATA_DIR=data/calib_v3 uv run --native-tls python -m arena_router.build_policy
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
from collections import Counter, defaultdict

import numpy as np

from arena_router.policy import Obs, Policy, arena_score, best_single, evaluate, fit
from arena_router.signals import CATEGORY_OF_SOURCE, CategoryClassifier, Embedder
from calib.common import DATA_DIR, ROOT

ARTIFACTS = ROOT / "artifacts"
EMB_CACHE = DATA_DIR / "body_embeddings.npz"
MODELS = "google/gemma-4-31b-it,openai/gpt-6-luna,google/gemini-3-flash-preview,deepseek/deepseek-v4.1-flash,qwen/qwen3-235b-a22b-2507,deepseek/deepseek-v4-flash-0731,deepseek/deepseek-v4.1-flash@low,google/gemini-3-flash-preview@off,deepseek/deepseek-v4-flash-0731@low"


def load_embeddings(prompts: list[dict]) -> dict[str, np.ndarray]:
    ids = [p["id"] for p in prompts]
    if EMB_CACHE.exists():
        d = np.load(EMB_CACHE)
        if list(d["ids"]) == ids:
            return dict(zip(ids, d["X"]))
    X = Embedder().encode([p["prompt"] for p in prompts])
    np.savez(EMB_CACHE, ids=np.array(ids), X=X)
    return dict(zip(ids, X))


def load_obs(models: list[str]) -> tuple[list[Obs], np.ndarray, list[str], dict]:
    """Obs(task=정답 범주, 이후 예측 범주로 교체), 본문 임베딩 행렬, 정답 범주."""
    rows = {r["id"]: r for r in map(json.loads, (DATA_DIR / "calib_set.jsonl").read_text().splitlines())}
    prompts = [json.loads(l) for l in (DATA_DIR / "calib_prompts.jsonl").read_text().splitlines()]
    emb = load_embeddings(prompts)
    n_source = Counter(r["config_name"] for r in rows.values())

    res: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for rec in map(json.loads, (DATA_DIR / "scored.jsonl").read_text().splitlines()):
        res[rec["id"]][rec["model"]] = (rec["accuracy"], rec["official_cost"])

    obs, X, y, missing = [], [], [], 0
    for id_, r in rows.items():
        if not all(m in res[id_] for m in models):
            missing += 1
            continue
        cat = CATEGORY_OF_SOURCE[r["config_name"]]
        obs.append(Obs(id_, cat, "", 1.0 / n_source[r["config_name"]], r["group_id"] or id_, res[id_]))
        X.append(emb[id_])
        y.append(cat)
    return obs, np.array(X), y, {"rows": len(rows), "missing_model_results": missing, "used": len(obs)}


def group_folds(obs: list[Obs], k: int, seed: int) -> list[list[int]]:
    groups = sorted({o.group_id for o in obs})
    random.Random(seed).shuffle(groups)
    fold_of = {g: i % k for i, g in enumerate(groups)}
    folds: list[list[int]] = [[] for _ in range(k)]
    for i, o in enumerate(obs):
        folds[fold_of[o.group_id]].append(i)
    return folds


def oof_categories(obs: list[Obs], X: np.ndarray, y: list[str], k: int, seed: int) -> list[str]:
    """각 문항의 범주를 그 문항을 보지 않은 분류기로 예측 (group k-fold)."""
    pred = [""] * len(obs)
    for fold in group_folds(obs, k, seed):
        test = set(fold)
        tr = [i for i in range(len(obs)) if i not in test]
        clf = CategoryClassifier.fit(X[tr], [y[i] for i in tr])
        for i, c in zip(fold, clf.predict(X[fold])):
            pred[i] = c
    return pred


def with_tasks(obs: list[Obs], tasks: list[str]) -> list[Obs]:
    return [dataclasses.replace(o, task=t) for o, t in zip(obs, tasks)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=MODELS)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    models = args.models.split(",")

    obs, X, y, load_info = load_obs(models)
    print(json.dumps(load_info))

    # 1) 바깥 group k-fold: fold마다 분류기(+안쪽 OOF 범주로 정책)와 최고 단일 모델을 train에서만 맞춘다
    held_router: dict[int, str] = {}
    held_single: dict[int, str] = {}
    held_cat: dict[int, str] = {}
    for fold in group_folds(obs, args.cv, args.seed):
        test = set(fold)
        tr = [i for i in range(len(obs)) if i not in test]
        tr_obs, tr_X, tr_y = [obs[i] for i in tr], X[tr], [y[i] for i in tr]
        p = fit(with_tasks(tr_obs, oof_categories(tr_obs, tr_X, tr_y, args.cv, args.seed + 1)), models, use_domain=False)
        clf = CategoryClassifier.fit(tr_X, tr_y)
        single = best_single(tr_obs, models)
        for i, c in zip(fold, clf.predict(X[fold])):
            held_cat[i] = c
            held_router[i] = p.route(c, "")
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
    cat_acc = float(np.mean([held_cat[i] == y[i] for i in all_idx]))

    # 2) 전체 데이터: OOF 범주로 정책, 전체로 분류기
    final = fit(with_tasks(obs, oof_categories(obs, X, y, args.cv, args.seed + 1)), models, use_domain=False)
    clf = CategoryClassifier.fit(X, y)
    true_obs = obs
    f_acc, f_cost, f_arena = evaluate(final, with_tasks(true_obs, clf.predict(X)))
    single_all = {m: evaluate(Policy(models, m), obs) for m in models}

    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "policy.json").write_text(json.dumps(final.to_dict(), indent=1))
    clf.save(ARTIFACTS / "category_clf.npz")
    usage = Counter(final.route(c, "") for c in clf.predict(X))
    report = {
        "load": load_info,
        "weights": "uniform per external source",
        "category_classifier_cv_accuracy": cat_acc,
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
    print(f"category classifier CV accuracy {cat_acc:.3f}")
    print(json.dumps(report["cv"], indent=1))
    print(json.dumps(report["final_in_sample"], indent=1))
    print("policy:", json.dumps(final.task_policy, indent=1), "default", final.default_model)
    for m, v in sorted(single_all.items(), key=lambda kv: -kv[1][2]):
        print(f"  single {m:34s} acc {v[0]*100:5.1f}  $/1K {v[1]:.3f}  arena {v[2]*100:5.2f}")


if __name__ == "__main__":
    main()
