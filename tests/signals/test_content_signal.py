"""내용 기반 신호: 지시문·답 형식 문단과 벤치마크 파일에 의존하지 않음을 확인한다."""
import json
import re
from pathlib import Path

import pytest

from arena_router.signals import CATEGORY_OF_SOURCE, question_body, routing_text

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
PROMPTS = ROOT / "data" / "calib_v3" / "calib_prompts.jsonl"


def test_body_drops_first_and_last_paragraph():
    assert question_body("Do X.\n\nQuestion: why?\n\nOptions: A) a\n\nAnswer in a box.") == "Question: why?\n\nOptions: A) a"


def test_format_cues_removed():
    p = "Please read.\n\nContext: None\n\nQuestion: How many legs?\n\nOptions:\nA) 4\nB) 6\n\nAnswer in a box."
    assert routing_text(p) == "How many legs?\n4\n6"


def test_relabelled_prompt_gives_same_routing_text():
    a = "Do this.\n\nQuestion: How many legs?\nOptions:\nA) 4\nB) 6\n\nBox it."
    b = "Something else entirely.\n\nQ: How many legs?\nChoices:\n(a) 4\n(b) 6\n\nReply briefly."
    assert routing_text(a) == routing_text(b)


def test_short_prompt_kept_whole():
    assert question_body("Translate this.\n\nHello") == "Translate this.\n\nHello"


def test_router_code_reads_no_benchmark_files():
    for f in ["signals.py", "policy.py", "router.py", "build_policy.py", "freeze.py"]:
        src = (ROOT / "arena_router" / f).read_text()
        assert not re.search(r"third_party|RouterArena/config|eval_config|targets\.json|routerarena_counts", src), f


def test_every_calibration_source_has_a_category():
    rows = ROOT / "data" / "calib_v3" / "calib_set.jsonl"
    if not rows.exists():
        pytest.skip("calibration set not built")
    srcs = {json.loads(l)["config_name"] for l in rows.read_text().splitlines()}
    assert srcs <= set(CATEGORY_OF_SOURCE)


@pytest.mark.skipif(not PROMPTS.exists() or not (ARTIFACTS / "category_clf.npz").exists(), reason="needs built artifacts")
def test_route_invariant_to_instruction_and_format_paragraphs():
    """첫 문단(지시문)과 마지막 문단(답 형식)을 임의 문장으로 바꿔도 라우팅이 같다."""
    from arena_router.router import ArenaRouter

    r = ArenaRouter()
    prompts = [json.loads(l)["prompt"] for l in PROMPTS.read_text().splitlines()][::97][:60]
    swapped = []
    for p in prompts:
        paras = p.strip().split("\n\n")
        if len(paras) >= 3:
            paras[0], paras[-1] = "Here is a task for you.", "Put the final answer in a box."
        swapped.append("\n\n".join(paras))
    assert [m for m, _ in r.route_batch(prompts)] == [m for m, _ in r.route_batch(swapped)]
