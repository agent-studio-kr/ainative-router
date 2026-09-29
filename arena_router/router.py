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
        raw = json.loads((artifacts / "policy.json").read_text())
        self.policy = Policy.from_dict(raw)
        self.classifier = CategoryClassifier.load(artifacts / "category_clf.npz")
        self.embedder = Embedder()
        # 선택: 범주 안에서 "기본 모델이 오래 생각할 것"으로 예측된 문항만 다른 선택지로 (arena_router.tail)
        self.tail_rule = raw.get("tail_rule")
        self.tail = None
        if self.tail_rule:
            from arena_router.tail import TailPredictor

            self.tail = TailPredictor.load(artifacts / "tail_model.npz")

    def route_batch(self, prompts: list[str]) -> list[tuple[str, dict]]:
        X = self.embedder.encode(prompts)
        cats = self.classifier.predict(X)
        out = [(self.policy.route(c, ""), {"category": c}) for c in cats]
        if self.tail is not None:
            rule = self.tail_rule
            idx = [i for i, c in enumerate(cats) if c == rule["task"]]
            if idx:
                pred = self.tail.predict([prompts[i] for i in idx], X[idx])
                for i, v in zip(idx, pred):
                    if v >= rule["threshold"]:
                        out[i] = (rule["model"], {"category": cats[i], "predicted_log_reasoning_tokens": float(v)})
        return out

    def route(self, prompt: str) -> str:
        return self.route_batch([prompt])[0][0]
