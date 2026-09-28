"""calib.sources.math 로더: 스키마 · RouterArena 누출 제외 · 프롬프트 생성 · RouterArena 채점기 정/오답 검증."""
import pytest

from calib.common import CalibRow, ExclusionIndex, format_prompt
from calib.scoring import score
from calib.sources.math import LOADERS, mathqa_options

N = 5


@pytest.fixture(scope="module")
def excl():
    return ExclusionIndex()


@pytest.mark.parametrize("config", sorted(LOADERS))
def test_loader(config, excl):
    rows = LOADERS[config](N, 0, excl)
    assert len(rows) == round(N * 1.3)
    assert [r.source_id for r in LOADERS[config](N, 0, excl)] == [r.source_id for r in rows]  # seed 결정적
    assert len({r.source_id for r in rows}) == len(rows)
    for r in rows:
        assert isinstance(r, CalibRow)
        assert r.config_name == r.dataset_name == config
        assert r.question and isinstance(r.answer, str) and r.answer
        assert r.source_hf.count(":") == 2 and r.source_id and r.group_id
        assert not excl.is_excluded(r.question, r.context)
        assert r.group_id not in excl.source_ids.get(config, set()) and r.source_id not in excl.source_ids.get(config, set())
        prompt = format_prompt(r)
        assert r.question.strip()[:30] in prompt.replace("{{", "{").replace("}}", "}")
        if config == "MathQA":
            assert len(r.options) >= 2 and r.answer in "abcde"
            right, wrong = r.answer.upper(), "ABCDE"[("abcde".index(r.answer) + 1) % 5]
        else:
            assert r.options == []
            right, wrong = r.answer, "-987654321"
        assert score(config, f"The answer is \\boxed{{{right}}}.", r.answer) == 1.0
        assert score(config, f"The answer is \\boxed{{{wrong}}}.", r.answer) == 0.0
    if config in ("AsDiv", "FinQA"):
        assert all(r.context for r in rows)
        assert "Context: None" not in format_prompt(rows[0])


def test_mathqa_options_match_routerarena_rule():
    # RouterArena 행에서 확인한 표현 (MathQA_84, MathQA_1003, MathQA_2157)
    assert mathqa_options("a ) 4 / 13 , b ) 1 / 13 , c ) 4 , d ) 1 , e ) 2 / 13") == ["4 / 13", "1 / 13", "4", "1", "2 / 13"]
    assert mathqa_options("a ) a ) 3 , b ) b ) 5 , c ) c ) 9 , d ) d ) 7 , e ) e ) 11") == [
        "a ) 3", "b ) 5", "c ) 9", "d ) 7", "e ) 11",
    ]
    assert mathqa_options("a ) 14,720 , b ) 16,240")[:3] == ["14", "720", "16"]


def test_aime_excludes_routerarena_problem_ids(excl):
    LOADERS["AIME"](N, 0, excl)
    ids = excl.source_ids["AIME"]
    assert len(ids) == len(excl.questions_by_dataset["AIME"]) == 39
    assert {"2022-I-1", "2025-I-4"} <= ids  # AIME_0, AIME_93
