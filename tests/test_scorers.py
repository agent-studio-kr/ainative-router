"""M2: RouterArena 공식 scorer 기대값 fixture. 정답/오답/빈 응답/형식 오류.

METEOR는 자체 구현이라 동일 문장도 1.0이 아니다 (~0.97) — 이 성질 자체를 고정해 둔다.
"""
import pytest

from calib.scoring import score

CASES = [
    # config, generated, gold, expected (exact) / ("gt", x) / ("lt", x)
    ("MMLUPro", "The answer is \\boxed{C}.", "C", 1.0),
    ("MMLUPro", "The answer is \\boxed{D}.", "C", 0.0),
    ("MMLUPro", "", "C", 0.0),
    ("AIME", "so \\boxed{116}", "116", 1.0),
    ("AIME", "so \\boxed{117}", "116", 0.0),
    ("MATH", "\\boxed{\\frac{1}{2}}", "\\frac{1}{2}", 1.0),
    ("MATH", "\\boxed{0.5}", "\\frac{1}{2}", 1.0),
    ("GSM8K", "\\boxed{18}", "18", 1.0),
    ("WMT19-de-en", "\\boxed{Das ist ein Test}", "Das ist ein Test", ("gt", 0.9)),
    ("WMT19-de-en", "\\boxed{Völlig anders}", "Das ist ein Test", ("lt", 0.3)),
]


@pytest.mark.parametrize("config,generated,gold,expected", CASES)
def test_scorer_fixture(config, generated, gold, expected):
    s = score(config, generated, gold)
    if isinstance(expected, tuple):
        op, x = expected
        assert (s > x) if op == "gt" else (s < x), s
    else:
        assert s == expected


def test_meteor_identical_is_not_one():
    assert score("WMT19-de-en", "\\boxed{Das ist ein Test}", "Das ist ein Test") < 1.0
