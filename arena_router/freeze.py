"""공식 평가 전 동결. 정책·범주 분류기·신호 코드·임베딩 모델 revision 의 해시를 기록/검증한다.

usage:
  uv run --native-tls python -m arena_router.freeze          # 동결 (FREEZE.json 기록)
  uv run --native-tls python -m arena_router.freeze --verify
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

from calib.common import ROOT

ARTIFACTS = ROOT / "artifacts"
CODE_FILES = ["arena_router/signals.py", "arena_router/policy.py", "arena_router/router.py"]
ARTIFACT_FILES = ["policy.json", "category_clf.npz", "category_clf.classes.json"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hf_revision(repo: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().model_info(repo).sha


def current_hashes(artifacts: Path = ARTIFACTS) -> dict:
    return {
        "code": {f: _sha(ROOT / f) for f in CODE_FILES},
        "artifacts": {f: _sha(artifacts / f) for f in ARTIFACT_FILES},
    }


def freeze(artifacts: Path = ARTIFACTS) -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import _tls  # noqa: F401

    from arena_router.signals import EMBED_MODEL

    record = {
        "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        **current_hashes(artifacts),
        "hf_revisions": {EMBED_MODEL: _hf_revision(EMBED_MODEL)},
        "note": "Frozen before routing RouterArena. Content-only routing (question body -> category -> model).",
    }
    (artifacts / "FREEZE.json").write_text(json.dumps(record, indent=1))
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
    ap.add_argument("--artifacts", default=str(ARTIFACTS), help="제출 버전별 산출물 디렉터리")
    args = ap.parse_args()
    if args.verify:
        verify(Path(args.artifacts))
        print("freeze OK")
    else:
        freeze(Path(args.artifacts))


if __name__ == "__main__":
    main()
