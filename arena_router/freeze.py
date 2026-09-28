"""M6: 공식 평가 전 동결. 정책·신호 코드·kNN 인덱스·외부 모델 revision 의 해시를 기록/검증한다.

usage:
  uv run --native-tls python -m arena_router.freeze          # 동결 (knn_index 생성 + FREEZE.json 기록)
  uv run --native-tls python -m arena_router.freeze --verify
  uv run --native-tls python -m arena_router.freeze --rebuild-index --verify   # 공개 레포 재현: 인덱스 재생성 후 해시 대조
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

from calib.common import DATA_DIR, ROOT

ARTIFACTS = ROOT / "artifacts"
CODE_FILES = ["arena_router/signals.py", "arena_router/policy.py", "arena_router/router.py"]
ARTIFACT_FILES = ["policy.json", "knn_index.jsonl"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hf_revision(repo: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().model_info(repo).sha


def build_knn_index() -> None:
    from arena_router.signals import KNN_PREFIX_CHARS

    sigs = {r["id"]: r for r in map(json.loads, (DATA_DIR / "signals.jsonl").read_text().splitlines())}
    with (ARTIFACTS / "knn_index.jsonl").open("w", encoding="utf-8") as f:
        for p in map(json.loads, (DATA_DIR / "calib_prompts.jsonl").read_text().splitlines()):
            f.write(json.dumps({"id": p["id"], "prompt_prefix": p["prompt"][:KNN_PREFIX_CHARS], "task": sigs[p["id"]]["task"]}, ensure_ascii=False) + "\n")


def current_hashes(artifacts: Path = ARTIFACTS) -> dict:
    return {
        "code": {f: _sha(ROOT / f) for f in CODE_FILES},
        "artifacts": {f: _sha(artifacts / f) for f in ARTIFACT_FILES},
    }


def freeze() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import _tls  # noqa: F401

    from arena_router.signals import DOMAIN_MODEL, EMBED_MODEL

    build_knn_index()
    record = {
        "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        **current_hashes(),
        "hf_revisions": {DOMAIN_MODEL: _hf_revision(DOMAIN_MODEL), EMBED_MODEL: _hf_revision(EMBED_MODEL)},
        "note": "RouterArena 공식 평가 전에 동결. 이후 정책/신호 변경 시 재동결 기록이 남는다.",
    }
    (ARTIFACTS / "FREEZE.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))


def verify(artifacts: Path = ARTIFACTS) -> None:
    rec = json.loads((artifacts / "FREEZE.json").read_text())
    now = current_hashes(artifacts)
    diff = [k for part in ("code", "artifacts") for k, v in now[part].items() if rec[part].get(k) != v]
    if diff:
        raise RuntimeError(f"freeze mismatch: {diff}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--rebuild-index", action="store_true", help="보정 세트에서 knn_index.jsonl 을 재생성 (FREEZE.json은 건드리지 않음)")
    args = ap.parse_args()
    if args.rebuild_index:
        build_knn_index()
    if args.verify:
        verify()
        print("freeze OK")
    elif not args.rebuild_index:
        freeze()


if __name__ == "__main__":
    main()
