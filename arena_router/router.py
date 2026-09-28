"""배포용 라우터: 동결된 정책 + 신호 → 모델명.

RouterArena 어댑터(third_party/RouterArena/router_inference/router/arena_router.py)가 이 클래스를 감싼다.
"""
from __future__ import annotations

import json
from pathlib import Path

from arena_router.policy import Policy
from arena_router.signals import SignalExtractor

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


class ArenaRouter:
    def __init__(self, artifacts: Path = ARTIFACTS) -> None:
        from arena_router.freeze import verify

        verify(artifacts)  # 동결 해시 불일치 시 예외
        self.policy = Policy.from_dict(json.loads((artifacts / "policy.json").read_text()))
        knn = [json.loads(l) for l in (artifacts / "knn_index.jsonl").read_text().splitlines()]
        self.signals = SignalExtractor([r["prompt_prefix"] for r in knn], [r["task"] for r in knn])

    def route_batch(self, prompts: list[str]) -> list[tuple[str, dict]]:
        out = []
        for s in self.signals.extract(prompts):
            out.append((self.policy.route(s.task, s.domain), s.__dict__))
        return out

    def route(self, prompt: str) -> str:
        return self.route_batch([prompt])[0][0]
