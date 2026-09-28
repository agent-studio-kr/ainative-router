"""M5: 셀 정책 로직 (합성 데이터)."""
from arena_router.policy import MIN_N, Obs, Policy, arena_score, evaluate, fit

M = ["cheap", "strong"]


def _obs(task, domain, n, cheap_acc, strong_acc, start=0):
    out = []
    for i in range(n):
        out.append(Obs(f"{task}-{domain}-{start+i}", task, domain, 1.0, f"g{task}{domain}{start+i}",
                       {"cheap": (float(i < cheap_acc * n), 0.0001), "strong": (float(i < strong_acc * n), 0.002)}))
    return out


def test_arena_score_matches_leaderboard():
    assert abs(arena_score(0.27, 0.7969) - 0.7762) < 1e-3  # Paix2 README 수치


def test_routes_hard_domain_to_strong_and_easy_to_cheap():
    obs = _obs("mcq", "math", 60, 0.3, 0.9) + _obs("mcq", "history", 60, 0.9, 0.92)
    p = fit(obs, M)
    assert p.route("mcq", "math") == "strong"
    assert p.route("mcq", "history") == "cheap"
    assert evaluate(p, obs)[2] > max(evaluate(Policy(M, m), obs)[2] for m in M)


def test_small_domain_cell_falls_back_to_task_choice():
    obs = _obs("mcq", "history", 80, 0.9, 0.92) + _obs("mcq", "law", MIN_N - 1, 0.0, 1.0)
    p = fit(obs, M)
    assert "law" not in p.task_domain_policy.get("mcq", {})
    assert p.route("mcq", "law") == p.task_policy["mcq"]
