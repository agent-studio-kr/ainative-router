"""Pre-registered holdout test: 11 content categories (A) vs categories × 3 predicted difficulty levels (B).

See docs/prereg_level_v4.md. Both arms are fitted on the exploration split of data/calib_v4 and scored once on the
confirmation split. No RouterArena data is used.

usage: HF_HUB_OFFLINE=1 CALIB_DATA_DIR=data/calib_v4 PYTHONPATH=. .venv/bin/python scripts/eval_prereg_level.py
"""
from __future__ import annotations

import dataclasses
import json
import random
import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

from arena_router.build_policy import group_folds, load_obs, oof_categories
from arena_router.policy import MIN_N, arena_score, fit
from arena_router.signals import CategoryClassifier, routing_text
from calib.common import DATA_DIR

M = ["google/gemma-4-31b-it", "google/gemini-3-flash-preview@off", "deepseek/deepseek-v4.1-flash@low",
     "deepseek/deepseek-v4-flash-0731@low", "deepseek/deepseek-v4-flash-0731"]
CHEAP = "google/gemma-4-31b-it"
LEVELS = 3
SPLIT_SEED = 20260930
N_BAG = 25
N_BOOT = 1000


def confirmation_ids() -> set[str]:
    """Group-level split stratified by external source: 40% of each source's groups go to confirmation."""
    rows = [json.loads(l) for l in (DATA_DIR / "calib_set.jsonl").read_text().splitlines()]
    by_src = defaultdict(set)
    for r in rows:
        by_src[r["config_name"]].add(r["group_id"] or r["id"])
    rng, hold = random.Random(SPLIT_SEED), set()
    for src in sorted(by_src):
        g = sorted(by_src[src])
        rng.shuffle(g)
        hold |= set(g[: round(len(g) * 0.4)])
    return {r["id"] for r in rows if (r["group_id"] or r["id"]) in hold}


def hand_raw(texts: list[str]) -> np.ndarray:
    """Surface features of the question body: length, digits, math symbols, LaTeX commands, line count."""
    out = []
    for t in texts:
        n = max(len(t), 1)
        out.append([np.log1p(len(t)), np.log1p(len(t.split())), sum(c.isdigit() for c in t) / n, t.count("$") / n * 100,
                    len(re.findall(r"\\[a-zA-Z]+", t)) / n * 100, t.count("\n") / 10, len(re.findall(r"[=+\-*/^]", t)) / n * 100])
    return np.array(out)


def difficulty_models(F, target, cats, obs):
    """Per-category logistic regression for P(cheap model wrong). Returns out-of-fold P on the fitting items and fitted models."""
    p_oof, models = np.full(len(obs), 0.5), {}
    cats = np.array(cats)
    folds = group_folds(obs, 5, 1)
    for c in sorted(set(cats)):
        idx = np.where(cats == c)[0]
        if len(idx) < 40 or len(set(target[idx])) < 2:
            continue
        for f in folds:
            te = np.intersect1d(f, idx)
            tr = np.setdiff1d(idx, te)
            if not len(te):
                continue
            if len(set(target[tr])) == 2:
                p_oof[te] = LogisticRegression(C=0.5, max_iter=3000).fit(F[tr], target[tr]).predict_proba(F[te])[:, 1]
            else:  # single-class training fold: constant prediction at the fold's positive rate
                p_oof[te] = target[tr].mean()
        models[c] = LogisticRegression(C=0.5, max_iter=3000).fit(F[idx], target[idx])
    return p_oof, models


def bagged_fit_cells(obs, n_bag=N_BAG, seed=0):
    """Category + sub-cell (domain) policy: sub-cells need MIN_N items and shrink to the category (policy.fit),
    then a group-bootstrap majority vote over n_bag refits."""
    base = fit(obs, M, use_domain=True)
    n_cell = Counter((o.task, o.domain) for o in obs)
    cells = {c for c, n in n_cell.items() if n >= MIN_N}  # eligibility fixed on the original sample, not per bootstrap
    rng = np.random.default_rng(seed)
    groups = sorted({o.group_id for o in obs})
    members = defaultdict(list)
    for i, o in enumerate(obs):
        members[o.group_id].append(i)
    vote_t, vote_c = defaultdict(Counter), defaultdict(Counter)
    for _ in range(n_bag):
        idx = [i for g in rng.choice(len(groups), len(groups)) for i in members[groups[g]]]
        p = fit([obs[i] for i in idx], M, use_domain=True)
        for t, d in cells:
            vote_c[(t, d)][p.route(t, d)] += 1
        for t in {t for t, _ in cells}:
            vote_t[t][p.task_policy.get(t, p.default_model)] += 1
    base.task_policy = {t: v.most_common(1)[0][0] for t, v in vote_t.items()}
    base.task_domain_policy = defaultdict(dict)  # ineligible sub-cells fall back to the category choice
    for (t, d), v in vote_c.items():
        base.task_domain_policy[t][d] = v.most_common(1)[0][0]
    return base


def main() -> None:
    obs, X, y, info = load_obs(M)
    hold_ids = confirmation_ids()
    prompts = {p["id"]: p["prompt"] for p in map(json.loads, (DATA_DIR / "calib_prompts.jsonl").read_text().splitlines())}
    H = hand_raw([routing_text(prompts[o.id]) for o in obs])
    dev = np.array([i for i, o in enumerate(obs) if o.id not in hold_ids])
    hold = np.array([i for i, o in enumerate(obs) if o.id in hold_ids])
    print(info, "exploration", len(dev), "confirmation", len(hold))

    mu, sd = H[dev].mean(0), H[dev].std(0) + 1e-9
    F = np.hstack([X, (H - mu) / sd])
    def reweight(ix):  # sources weighted equally within each split (counts from that split only)
        n = Counter(src[i] for i in ix)
        return [dataclasses.replace(obs[i], weight=1.0 / n[src[i]]) for i in ix]

    rows = {r["id"]: r for r in map(json.loads, (DATA_DIR / "calib_set.jsonl").read_text().splitlines())}
    src = [rows[o.id]["config_name"] for o in obs]
    dobs = reweight(dev)
    ydev = [y[i] for i in dev]
    cat_dev = oof_categories(dobs, X[dev], ydev, 5, 1)
    cat_hold = CategoryClassifier.fit(X[dev], ydev).predict(X[hold])

    target = np.array([obs[i].results[CHEAP][0] < 0.5 for i in dev]).astype(int)
    p_dev, dmodels = difficulty_models(F[dev], target, cat_dev, dobs)
    cuts = {c: np.quantile(p_dev[np.array(cat_dev) == c], np.linspace(0, 1, LEVELS + 1)[1:-1]) for c in dmodels}

    def level(p, c):
        return str(int(np.searchsorted(cuts[c], p))) if c in cuts else ""

    lvl_dev = [level(p, c) for p, c in zip(p_dev, cat_dev)]
    lvl_hold = [level(dmodels[c].predict_proba(F[[i]])[0, 1], c) if c in dmodels else "" for i, c in zip(hold, cat_hold)]

    pol_a = bagged_fit_cells([dataclasses.replace(o, task=c, domain="") for o, c in zip(dobs, cat_dev)])
    pol_b = bagged_fit_cells([dataclasses.replace(o, task=c, domain=l) for o, c, l in zip(dobs, cat_dev, lvl_dev)])
    hobs = reweight(hold)
    ch_a = [pol_a.route(c, "") for c in cat_hold]
    ch_b = [pol_b.route(c, l) for c, l in zip(cat_hold, lvl_hold)]

    w = np.array([o.weight for o in hobs])

    def agg(idx, ch):
        a = np.array([hobs[i].results[ch[i]][0] for i in idx])
        cst = np.array([hobs[i].results[ch[i]][1] for i in idx])
        acc, cost = (w[idx] * a).sum() / w[idx].sum(), (w[idx] * cst).sum() / w[idx].sum() * 1000
        return acc, cost, arena_score(cost, acc)

    allidx = np.arange(len(hobs))
    for name, ch in [("A (11 categories)", ch_a), ("B (categories x 3 levels)", ch_b)]:
        acc, cost, s = agg(allidx, ch)
        print(f"{name}: accuracy {acc * 100:.2f}%  ${cost:.3f}/1K  Arena {s * 100:.2f}")
    groups = defaultdict(list)
    for i, o in enumerate(hobs):
        groups[o.group_id].append(i)
    gk = list(groups)
    rng = np.random.default_rng(0)
    deltas = []
    for _ in range(N_BOOT):
        idx = np.array([i for g in rng.choice(len(gk), len(gk)) for i in groups[gk[g]]])
        deltas.append(agg(idx, ch_b)[2] - agg(idx, ch_a)[2])
    lo, med, hi = np.percentile(deltas, [2.5, 50, 97.5]) * 100
    print(f"Delta Arena (B - A) {med:+.2f} [{lo:+.2f}, {hi:+.2f}] -> {'ADOPT B' if lo > 0 else 'KEEP A'}")
    print("B sub-cells that differ from the category choice:",
          {t: {d: m for d, m in ds.items() if m != pol_b.task_policy.get(t)} for t, ds in pol_b.task_domain_policy.items()
           if any(m != pol_b.task_policy.get(t) for m in ds.values())})


if __name__ == "__main__":
    main()
