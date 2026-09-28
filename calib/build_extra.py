"""보정 세트 확장: 기존 세트(data/calib)와 RouterArena를 모두 제외한 추가 문항을 뽑아 data/calib_v2 에 병합본을 만든다.

- 기존 파일(v1 매니페스트 재현용)은 건드리지 않는다.
- 추가분 id 접두사: cal2_
usage: uv run --native-tls python -m calib.build_extra --seed 11 --scale 1.0
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from calib.build import GROUPS, select
from calib.common import ROOT, ExclusionIndex, format_prompt, normalize

OLD = ROOT / "data" / "calib"
NEW = ROOT / "data" / "calib_v2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--scale", type=float, default=1.0, help="config별 추가 목표 = 기존 목표 × scale")
    ap.add_argument("--out", default=str(NEW), help="출력 디렉터리 (재현 검증용)")
    args = ap.parse_args()
    out = Path(args.out)

    out.mkdir(parents=True, exist_ok=True)
    if (OLD / "cache").exists() and not (out / "cache").exists():
        shutil.copytree(OLD / "cache", out / "cache")

    old_rows = [json.loads(l) for l in (OLD / "calib_set.jsonl").read_text().splitlines()]
    seen_src = {(r["source_hf"], r["source_id"]) for r in old_rows if r["source_id"]}
    seen_q = {normalize(r["question"]) + "|" + normalize(r.get("context", ""))[:200] for r in old_rows}

    targets = json.loads((ROOT / "calib" / "targets.json").read_text())["by_config"]
    loaders = {}
    for g in GROUPS:
        loaders.update(g.LOADERS)
    excl = ExclusionIndex()
    rng = random.Random(args.seed)

    extra, report = [], {}
    for cfg in sorted(loaders):
        target = round(targets.get(cfg, 0) * args.scale)
        if target == 0:
            continue
        # 기존 세트와 겹치지 않는 후보를 충분히 얻기 위해 넉넉히 요청
        cands = loaders[cfg](target * 3, args.seed, excl)
        fresh = []
        for r in cands:
            key = normalize(r.question) + "|" + normalize(r.context)[:200]
            if (r.source_hf, r.source_id) in seen_src or key in seen_q or excl.is_excluded(r.question, r.context):
                continue
            seen_q.add(key)
            fresh.append(r)
        chosen = select(fresh, target, rng)
        extra += chosen
        report[cfg] = {"target": target, "fresh_candidates": len(fresh), "selected": len(chosen)}
        print(f"{cfg:28s} target {target:4d} fresh {len(fresh):5d} sel {len(chosen):4d}", flush=True)

    ids = [f"cal2_{r.config_name}_{i:05d}" for i, r in enumerate(extra)]
    # 병합본: 기존 + 추가
    with (out / "calib_set.jsonl").open("w", encoding="utf-8") as f, (out / "lcb_answers.jsonl").open("w", encoding="utf-8") as g:
        for line in (OLD / "calib_set.jsonl").read_text().splitlines():
            f.write(line + "\n")
        for line in (OLD / "lcb_answers.jsonl").read_text().splitlines():
            g.write(line + "\n")
        for id_, r in zip(ids, extra):
            rec = {"id": id_, **r.to_json()}
            if r.config_name == "LiveCodeBench":
                g.write(json.dumps({"id": id_, "answer": rec["answer"]}, ensure_ascii=False) + "\n")
                rec["answer"] = {"_ref": "lcb_answers.jsonl"}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (out / "calib_prompts.jsonl").open("w", encoding="utf-8") as f:
        for line in (OLD / "calib_prompts.jsonl").read_text().splitlines():
            f.write(line + "\n")
        for id_, r in zip(ids, extra):
            f.write(json.dumps({"id": id_, "config_name": r.config_name, "prompt": format_prompt(r)}, ensure_ascii=False) + "\n")
    (out / "build_extra_report.json").write_text(json.dumps({"added": len(extra), "configs": report}, indent=1))
    print(f"added {len(extra)} rows → {out} (total {len(old_rows) + len(extra)})")


if __name__ == "__main__":
    main()
