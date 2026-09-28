"""재생성한 보정 세트가 공개 매니페스트와 같은지 확인 (id, 원천, 프롬프트 SHA-256)."""
import hashlib
import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "calib"
manifest = {r["id"]: r for r in map(json.loads, (DATA / "calib_manifest.jsonl").read_text().splitlines())}
prompts = {r["id"]: r["prompt"] for r in map(json.loads, (DATA / "calib_prompts.jsonl").read_text().splitlines())}
bad = [i for i, m in manifest.items() if i not in prompts or hashlib.sha256(prompts[i].encode()).hexdigest() != m["prompt_sha256"]]
extra = set(prompts) - set(manifest)
print(f"manifest {len(manifest)}, rebuilt {len(prompts)}, mismatched {len(bad)}, extra {len(extra)}")
sys.exit(1 if bad or extra else 0)
