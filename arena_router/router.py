"""배포용 라우터: 질문 본문 → 내용 범주 → 동결된 정책의 모델.

RouterArena 어댑터가 이 클래스를 감싼다. 벤치마크 파일은 읽지 않는다.
"""
from __future__ import annotations

import json
from pathlib import Path

from arena_router.policy import Policy
from arena_router.signals import CategoryClassifier, Embedder

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


class ArenaRouter:
    def __init__(self, artifacts: Path = ARTIFACTS) -> None:
        from arena_router.freeze import verify

        verify(artifacts)  # 동결 해시 불일치 시 예외
        self.policy = Policy.from_dict(json.loads((artifacts / "policy.json").read_text()))
        self.classifier = CategoryClassifier.load(artifacts / "category_clf.npz")
        self.embedder = Embedder()

    def route_batch(self, prompts: list[str]) -> list[tuple[str, dict]]:
        cats = self.classifier.predict(self.embedder.encode(prompts))
        return [(self.policy.route(c, ""), {"category": c}) for c in cats]

    def route(self, prompt: str) -> str:
        return self.route_batch([prompt])[0][0]
