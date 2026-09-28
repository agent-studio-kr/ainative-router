"""셀 → 모델 정책 (셀별 모델 배정 + λ 스윕으로 Arena Score 직접 최적화).

셀 = 질문 본문에서 예측한 내용 범주 (arena_router.signals). 도메인 하위 셀은 쓰지 않는다(use_domain=False).
- 셀 추정치는 전체로 수축(shrinkage): est = (n·mean + K0·parent) / (n + K0)
- λ 스윕: 셀마다 argmax(acc − λ·cost), 가중 Arena Score가 최대인 λ 채택
- 가중치: 외부 원천마다 합이 1 (원천 균등)
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

K0 = 10.0
MIN_N = 20
LAMBDAS = [0.0] + list(np.geomspace(1, 3000, 60))  # 비용 단위: USD/query → λ는 정확도 1 대비 USD 가치의 역수


def arena_score(cost_per_1k: float, acc: float, beta: float = 0.1, c_max: float = 200.0, c_min: float = 0.0044) -> float:
    """RouterArena llm_evaluation/run.py compute_arena_score 와 동일."""
    cost = max(c_min, min(cost_per_1k, c_max))
    c = (math.log2(c_max) - math.log2(cost)) / (math.log2(c_max) - math.log2(c_min))
    return (1 + beta) * acc * c / (beta * acc + c)


@dataclass
class Obs:
    """한 문항에 대한 모델별 결과 (acc, cost) + 신호 + 가중치."""
    id: str
    task: str
    domain: str
    weight: float
    group_id: str
    results: dict[str, tuple[float, float]]  # model -> (accuracy, official_cost USD)


@dataclass
class Policy:
    models: list[str]
    default_model: str
    task_policy: dict[str, str] = field(default_factory=dict)
    task_domain_policy: dict[str, dict[str, str]] = field(default_factory=dict)
    lam: float = 0.0

    def route(self, task: str, domain: str) -> str:
        by_dom = self.task_domain_policy.get(task)
        if by_dom and domain in by_dom:
            return by_dom[domain]
        return self.task_policy.get(task, self.default_model)

    def to_dict(self) -> dict:
        return {
            "models": self.models,
            "default_model": self.default_model,
            "lambda": float(self.lam),
            "task_policy": dict(sorted(self.task_policy.items())),
            "task_domain_policy": {t: dict(sorted(d.items())) for t, d in sorted(self.task_domain_policy.items())},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Policy":
        return cls(d["models"], d["default_model"], d["task_policy"], d["task_domain_policy"], d["lambda"])


def _cell_stats(obs: list[Obs], models: list[str]) -> tuple[float, dict[str, tuple[float, float]]]:
    """가중 평균 (acc, cost) per model. 반환: (n, {model: (acc, cost)})"""
    wsum = sum(o.weight for o in obs)
    out = {}
    for m in models:
        out[m] = (
            sum(o.weight * o.results[m][0] for o in obs) / wsum,
            sum(o.weight * o.results[m][1] for o in obs) / wsum,
        )
    return float(len(obs)), out


def _shrink(n: float, stats: dict, parent: dict) -> dict:
    return {
        m: ((n * a + K0 * parent[m][0]) / (n + K0), (n * c + K0 * parent[m][1]) / (n + K0))
        for m, (a, c) in stats.items()
    }


def _pick(est: dict, lam: float) -> str:
    return max(est, key=lambda m: est[m][0] - lam * est[m][1])


def evaluate(policy: Policy, obs: list[Obs]) -> tuple[float, float, float]:
    """가중 (accuracy, cost_per_1k, arena)."""
    wsum = sum(o.weight for o in obs)
    acc = sum(o.weight * o.results[policy.route(o.task, o.domain)][0] for o in obs) / wsum
    cost = sum(o.weight * o.results[policy.route(o.task, o.domain)][1] for o in obs) / wsum * 1000
    return acc, cost, arena_score(cost, acc)


def best_single(obs: list[Obs], models: list[str]) -> str:
    return max(models, key=lambda m: evaluate(Policy(models, m), obs)[2])


def fit(obs: list[Obs], models: list[str], use_domain: bool = False) -> Policy:
    _, g_stats = _cell_stats(obs, models)
    by_task: dict[str, list[Obs]] = defaultdict(list)
    for o in obs:
        by_task[o.task].append(o)

    task_est: dict[str, dict] = {}
    dom_est: dict[str, dict[str, dict]] = defaultdict(dict)
    for t, tobs in by_task.items():
        n, st = _cell_stats(tobs, models)
        task_est[t] = _shrink(n, st, g_stats)
        if use_domain:
            by_dom: dict[str, list[Obs]] = defaultdict(list)
            for o in tobs:
                by_dom[o.domain].append(o)
            if len(by_dom) > 1:
                for d, dobs in by_dom.items():
                    if len(dobs) >= MIN_N:
                        dn, dst = _cell_stats(dobs, models)
                        dom_est[t][d] = _shrink(dn, dst, task_est[t])

    best: Policy | None = None
    best_score = -1.0
    for lam in LAMBDAS:
        p = Policy(models, _pick(g_stats, lam), lam=lam)
        p.task_policy = {t: _pick(e, lam) for t, e in task_est.items()}
        p.task_domain_policy = {
            t: {d: m for d, e in doms.items() if (m := _pick(e, lam)) != p.task_policy[t]} for t, doms in dom_est.items()
        }
        p.task_domain_policy = {t: d for t, d in p.task_domain_policy.items() if d}
        score = evaluate(p, obs)[2]
        if score > best_score:
            best, best_score = p, score
    assert best is not None
    return best
