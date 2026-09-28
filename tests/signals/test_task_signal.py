"""M4: 과제 신호 정확도 (보정 세트 기준, 외부 데이터만 사용)."""
import json
import random

from arena_router.signals import SignalExtractor, template_heads
from calib.common import DATA_DIR


def _load():
    rows = [json.loads(l) for l in (DATA_DIR / "calib_prompts.jsonl").read_text().splitlines()]
    cfg2group = {c: g for g in template_heads().values() for c in g.split("|")}
    return rows, cfg2group


def test_template_task_accuracy():
    rows, cfg2group = _load()
    ex = SignalExtractor([], [])
    wrong = [r["id"] for r in rows if ex.task_by_template(r["prompt"]) != cfg2group[r["config_name"]]]
    assert len(wrong) / len(rows) <= 0.01, wrong[:10]


def test_knn_fallback_on_paraphrased_template():
    """지시문 머리말을 LLM 패러프레이즈로 바꾼 프롬프트에서 kNN 폴백 정확도 (robustness split 유형 모사).

    인덱스와 평가를 문항 단위로 반씩 나눈다. 패러프레이즈: data/calib/template_paraphrases.json
    """
    rows, cfg2group = _load()
    paras = json.loads((DATA_DIR / "template_paraphrases.json").read_text())
    by_group = {g: v for g, v in paras.items()}
    random.Random(0).shuffle(rows)
    half = len(rows) // 2
    index, held = rows[:half], rows[half:]
    ex = SignalExtractor([r["prompt"] for r in index], [cfg2group[r["config_name"]] for r in index])
    rng = random.Random(1)
    prompts, gold = [], []
    for r in held:
        g = cfg2group[r["config_name"]]
        cands = [(h["head"], p) for h in by_group.get(g, []) for p in h["paraphrases"] if r["prompt"].lstrip().startswith(h["head"])]
        if not cands:
            continue
        head, para = rng.choice(cands)
        prompts.append(r["prompt"].lstrip().replace(head, para, 1))
        gold.append(g)
    assert all(ex.task_by_template(p) is None for p in prompts[:50])  # 템플릿 매칭이 실제로 깨졌는지
    pred = ex.task_by_knn(prompts)
    acc = sum(p == g for p, g in zip(pred, gold)) / len(gold)
    print(f"knn fallback accuracy on paraphrased templates: {acc:.3f} (n={len(gold)})")
    assert acc >= 0.9
