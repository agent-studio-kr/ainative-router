"""M1: 원천 로더를 모아 보정 세트를 만든다.

출력:
  data/calib/calib_set.jsonl      CalibRow + id
  data/calib/calib_prompts.jsonl  {id, config_name, prompt}  (calib.infer 입력)
  data/calib/build_report.json    config별 후보 수 / 선택 수 / 목표

usage: uv run --native-tls python -m calib.build [--seed 7]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

from calib.common import DATA_DIR, ROOT, CalibRow, ExclusionIndex, config_manifest, format_prompt, normalize
from calib.sources import code_chess, knowledge, language, math

GROUPS = [knowledge, math, language, code_chess]
LCB_ANSWERS = "lcb_answers.jsonl"


def select(rows: list[CalibRow], target: int, rng: random.Random) -> list[CalibRow]:
    """subset(dataset_name) 비율을 유지하며 target개 선택."""
    if len(rows) <= target:
        return rows
    by_subset: dict[str, list[CalibRow]] = defaultdict(list)
    for r in rows:
        by_subset[r.dataset_name].append(r)
    picked: list[CalibRow] = []
    for subset, items in sorted(by_subset.items()):
        rng.shuffle(items)
        picked += items[: max(1, round(target * len(items) / len(rows)))]
    rng.shuffle(picked)
    return picked[:target]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    targets = json.loads((ROOT / "calib" / "targets.json").read_text())["by_config"]
    loaders = {}
    for g in GROUPS:
        dup = loaders.keys() & g.LOADERS.keys()
        assert not dup, f"duplicate loaders: {dup}"
        loaders.update(g.LOADERS)
    missing = set(config_manifest()) - loaders.keys()
    excl = ExclusionIndex()
    rng = random.Random(args.seed)

    rows: list[CalibRow] = []
    report = {"missing_loaders": sorted(missing), "configs": {}}
    for cfg in sorted(loaders):
        target = targets.get(cfg, 0)
        if target == 0:
            continue
        cands = loaders[cfg](target, args.seed, excl)
        bad = [r for r in cands if r.config_name != cfg or excl.is_excluded(r.question, r.context)]
        assert not bad, f"{cfg}: {len(bad)} invalid/excluded rows returned"
        chosen = select(cands, target, rng)
        rows += chosen
        report["configs"][cfg] = {"target": target, "candidates": len(cands), "selected": len(chosen)}
        print(f"{cfg:28s} target {target:4d} cand {len(cands):5d} sel {len(chosen):4d}", flush=True)

    # 보정 세트 내부 중복 질문 제거 (예: MMLU-Pro에 섞인 원본 MMLU 문항 ↔ ArcMMLU 대체 MMLU 문항)
    seen: set[str] = set()
    deduped = []
    for r in rows:
        key = normalize(r.question) + "|" + normalize(r.context)[:200]
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    report["intra_duplicates_removed"] = len(rows) - len(deduped)
    rows = deduped

    ids = [f"cal_{r.config_name}_{i:05d}" for i, r in enumerate(rows)]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # LiveCodeBench 정답(테스트 케이스)은 문항당 최대 수십 MB라 본문과 분리 저장한다
    with (DATA_DIR / "calib_set.jsonl").open("w", encoding="utf-8") as f, (DATA_DIR / LCB_ANSWERS).open(
        "w", encoding="utf-8"
    ) as g:
        for id_, r in zip(ids, rows):
            rec = {"id": id_, **r.to_json()}
            if r.config_name == "LiveCodeBench":
                g.write(json.dumps({"id": id_, "answer": rec["answer"]}, ensure_ascii=False) + "\n")
                rec["answer"] = {"_ref": LCB_ANSWERS}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (DATA_DIR / "calib_prompts.jsonl").open("w", encoding="utf-8") as f:
        for id_, r in zip(ids, rows):
            f.write(json.dumps({"id": id_, "config_name": r.config_name, "prompt": format_prompt(r)}, ensure_ascii=False) + "\n")
    report["total"] = len(rows)
    (DATA_DIR / "build_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"total {len(rows)} rows; missing loaders: {sorted(missing)}")


if __name__ == "__main__":
    main()
