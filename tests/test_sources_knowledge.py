"""calib/sources/knowledge.py 로더 검증: 스키마 · 누출 제외 · RouterArena 프롬프트/채점기 호환 · 목표 수량 · 결정성."""
import json
import math

import pytest

from calib.common import ROOT, CalibRow, ExclusionIndex, config_for, format_prompt
from calib.scoring import score
from calib.sources.knowledge import LETTERS, LOADERS

TARGETS = json.loads((ROOT / "calib" / "targets.json").read_text())["by_config"]
CONFIGS = sorted(LOADERS)


@pytest.fixture(scope="module")
def excl() -> ExclusionIndex:
    return ExclusionIndex()


def _gold_letter(row: CalibRow) -> str:
    return row.answer if row.answer in LETTERS else LETTERS[int(row.answer)]


@pytest.mark.parametrize("config", CONFIGS)
def test_small_load_schema_exclusion_prompt_score(config, excl):
    rows = LOADERS[config](5, 0, excl)
    assert 0 < len(rows) <= math.ceil(5 * 1.3)
    for row in rows:
        assert isinstance(row, CalibRow)
        assert row.config_name == config
        assert config_for(row.dataset_name, bool(row.options)) == config
        assert row.question.strip() and len(row.options) >= 2
        assert isinstance(row.answer, str)
        assert row.source_hf and row.source_id and row.group_id
        assert not excl.is_excluded(row.question, row.context)
        assert row.source_id not in excl.source_ids.get(config, set())

        prompt = format_prompt(row)
        assert row.question[:200] in prompt
        for i, opt in enumerate(row.options):
            assert f"{LETTERS[i]}. {opt}" in prompt

        gold = _gold_letter(row)
        wrong = next(LETTERS[i] for i in range(len(row.options)) if LETTERS[i] != gold)
        assert score(config, f"Reasoning... \\boxed{{{gold}}}", row.answer) == 1.0
        assert score(config, f"Reasoning... \\boxed{{{wrong}}}", row.answer) == 0.0


@pytest.mark.parametrize("config", CONFIGS)
def test_full_target_count_and_determinism(config, excl):
    target = TARGETS[config]
    rows = LOADERS[config](target, 0, excl)
    assert target <= len(rows) <= math.ceil(target * 1.3)
    assert len({r.source_id for r in rows}) == len(rows)
    assert [r.source_id for r in LOADERS[config](target, 0, excl)] == [r.source_id for r in rows]
    assert not any(excl.is_excluded(r.question, r.context) for r in rows)
