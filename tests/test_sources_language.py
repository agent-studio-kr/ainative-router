"""calib.sources.language 로더 검증: 스키마 · RouterArena 제외 · 프롬프트 생성 · 공식 채점기로 정답/오답 점수."""
import pytest

from calib.common import ExclusionIndex, config_for, format_prompt
from calib.scoring import score
from calib.sources.language import LOADERS

TARGET = 5
WRONG_SENTENCE = "The quick brown fox jumps over the lazy dog near the old river bank."


@pytest.fixture(scope="module")
def excl():
    return ExclusionIndex()


def _boxed(x: str) -> str:
    return f"The final answer is \\boxed{{{x}}}."


def _gold_and_wrong(cfg: str, answer: str) -> tuple[str, str, float]:
    """(정답 출력, 오답 출력, 정답 최소 점수)."""
    # RouterArena METEOR 구현은 동일 문자열도 1.0 미만(조각 페널티): 문장은 ~0.97+, 1단어 정답은 0.875 가 상한
    if cfg.startswith("WMT19"):
        return _boxed(answer), _boxed(WRONG_SENTENCE), 0.9
    if cfg == "NarrativeQA":
        return _boxed(answer), _boxed(WRONG_SENTENCE), 0.85
    if cfg == "SuperGLUE-CausalReasoning":
        i = int(float(answer))
        return _boxed("AB"[i]), _boxed("AB"[1 - i]), 1.0
    if cfg in ("SuperGLUE-Entailment", "SuperGLUE-QA", "SuperGLUE-RC"):
        i = int(float(answer))
        return _boxed(str(i)), _boxed(str(1 - i)), 1.0
    if cfg in ("SuperGLUE-Wic", "SuperGLUE-Wsc"):
        return _boxed(answer), _boxed("No" if answer == "Yes" else "Yes"), 1.0
    if cfg == "QANTA":
        return _boxed(answer.replace("_", " ")), _boxed("Zzqx Nonexistent"), 1.0
    return _boxed(answer), _boxed("Zzqx Nonexistent"), 1.0  # ClozeTest, GeoGraphyData


@pytest.mark.parametrize("cfg", sorted(LOADERS))
def test_loader(cfg, excl):
    rows = LOADERS[cfg](TARGET, 0, excl)
    assert TARGET <= len(rows) <= 7, len(rows)
    again = LOADERS[cfg](TARGET, 0, excl)
    assert [r.source_id for r in rows] == [r.source_id for r in again], "seed 결정성"
    assert len({r.source_id for r in rows}) == len(rows)

    blocked = excl.source_ids.get(cfg, set())  # 로더가 채운 RouterArena 원천 ID/지문 키 (config 별)
    for r in rows:
        # 스키마
        assert r.config_name == cfg
        assert config_for(r.dataset_name, bool(r.options)) == cfg
        assert isinstance(r.question, str) and r.question.strip()
        assert isinstance(r.answer, str) and r.answer.strip()
        assert all(isinstance(o, str) for o in r.options)
        assert r.source_hf and r.source_id and r.group_id
        # RouterArena 제외
        if cfg == "SuperGLUE-CausalReasoning":  # 질문이 고정 템플릿 → 전제문(Context) 단위 제외
            assert not excl.is_excluded(r.context)
        else:
            assert not excl.is_excluded(r.question, r.context)
        assert r.source_id not in blocked and r.group_id not in blocked
        # 프롬프트
        prompt = format_prompt(r)
        assert "{Question}" not in prompt and r.question.strip()[:30] in prompt
        # 채점기
        gold, wrong, floor = _gold_and_wrong(cfg, r.answer)
        assert score(cfg, gold, r.answer) >= floor, (r.answer, score(cfg, gold, r.answer))
        assert score(cfg, wrong, r.answer) < 0.5, (r.answer, score(cfg, wrong, r.answer))


def test_qanta_subsets_proportional(excl):
    rows = LOADERS["QANTA"](153, 0, excl)
    names = {r.dataset_name for r in rows}
    assert names == {"QANTA_Fine Arts", "QANTA_Geography", "QANTA_History", "QANTA_Literature",
                     "QANTA_Philosophy", "QANTA_Science", "QANTA_Social Science"}
    lit = sum(r.dataset_name == "QANTA_Literature" for r in rows)
    assert abs(lit / len(rows) - 221 / 644) < 0.03
