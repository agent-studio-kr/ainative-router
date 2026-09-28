"""calib.common.format_prompt 가 RouterArena 실제 프롬프트와 바이트 단위로 같은지 검증 (LiveCodeBench 제외)."""
import json

from calib.common import ROUTERARENA_DIR, CalibRow, config_for, format_prompt


def test_format_prompt_matches_routerarena():
    from datasets import load_dataset

    full = {r["Global Index"]: r for r in load_dataset("RouteWorks/RouterArena", split="full")}
    preds = json.loads((ROUTERARENA_DIR / "router_inference/predictions/vllm-sr.json").read_text())
    checked, mismatches = 0, []
    for p in preds:
        if p.get("for_optimality"):
            continue
        r = full[p["global index"]]
        if r["Dataset name"] == "LiveCodeBench":
            continue
        opts = r["Options"]
        opts = json.loads(opts) if isinstance(opts, str) else (opts or [])
        row = CalibRow(
            dataset_name=r["Dataset name"],
            config_name=config_for(r["Dataset name"], bool(opts)),
            question=r["Question"],
            answer=r["Answer"],
            context=r["Context"],
            options=opts,
        )
        checked += 1
        if format_prompt(row) != p["prompt"]:
            mismatches.append(p["global index"])
    assert checked > 7000
    assert not mismatches, f"{len(mismatches)} mismatches, e.g. {mismatches[:5]}"
