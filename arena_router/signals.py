"""라우팅 신호: 질문 본문의 내용 범주 (content category).

- 본문 추출은 구조 규칙만 쓴다: 빈 줄로 나눈 문단 중 첫 문단(지시문)과 마지막 문단(답 형식)을 뗀 뒤,
  줄 머리의 짧은 라벨("Xxx:"), 보기 기호("A)"), 빈 자리표시 줄("None")을 지운다(형식 단서 제거).
  벤치마크 설정 파일이나 지시문 문자열은 읽지도, 비교하지도 않는다.
- 범주 분류기: all-MiniLM-L6-v2 본문 임베딩 → 다항 로지스틱 회귀. 외부 보정 문항으로만 학습하며,
  라벨은 각 문항의 외부 원천을 우리 범주 체계(CATEGORY_OF_SOURCE)로 옮긴 것이다.
"""
from __future__ import annotations

import json
import re
from functools import cached_property
from pathlib import Path

import numpy as np

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MIN_PARAGRAPHS = 3

# 외부 원천 → 내용 범주 (학습 라벨 전용; 라우팅 시에는 쓰지 않는다)
CATEGORY_OF_SOURCE = {
    **{s: "mcq_knowledge" for s in [
        "ArcMMLU", "GeoBench", "MMLU", "MMLUPro", "MedMCQA", "MusicTheoryBench", "OpenTDB", "PubMedQA",
        "SocialiQA", "SuperGLUE-CausalReasoning", "GPQA",
    ]},
    "MathQA": "mcq_math",
    **{s: "math" for s in ["GSM8K", "MATH", "AsDiv", "AIME"]},
    "FinQA": "finance_math",
    "LiveCodeBench": "code",
    **{f"WMT19-{p}-en": "translation" for p in ["cs", "de", "fi", "gu", "kk", "lt", "ru", "zh"]},
    **{s: "reading" for s in ["NarrativeQA", "SuperGLUE-QA", "SuperGLUE-RC", "SuperGLUE-ClozeTest"]},
    **{s: "nli" for s in ["SuperGLUE-Entailment", "SuperGLUE-Wic", "SuperGLUE-Wsc"]},
    **{s: "ethics" for s in ["Ethics_commonsense", "Ethics_deontology", "Ethics_justice", "Ethics_virtue"]},
    **{s: "chess" for s in ["ChessInstruct", "ChessInstruct_mcq"]},
    **{s: "trivia" for s in ["QANTA", "GeoGraphyData"]},
}


def question_body(prompt: str) -> str:
    """첫 문단(지시문)과 마지막 문단(답 형식)을 뗀 본문. 문단이 3개 미만이면 원문."""
    paras = [p for p in prompt.strip().split("\n\n") if p.strip()]
    if len(paras) < MIN_PARAGRAPHS:
        return prompt.strip()
    return "\n\n".join(paras[1:-1]).strip()


_LABEL = re.compile(r"(?m)^[ \t]*[A-Za-z][A-Za-z0-9 _\-\"']{0,30}:[ \t]*")
_OPTION = re.compile(r"(?m)^[ \t]*\(?[A-Ja-j][\).:\]][ \t]+")
_PLACEHOLDER = re.compile(r"(?mi)^[ \t]*(none|null|n/a)[ \t]*$")


def normalize_body(body: str) -> str:
    """형식 단서 제거: 줄 머리 라벨, 보기 기호, 빈 자리표시 줄. 남는 것은 질문·지문·보기의 내용뿐."""
    text = _PLACEHOLDER.sub("", _OPTION.sub("", _LABEL.sub("", body)))
    return re.sub(r"\n{2,}", "\n", text).strip()


def routing_text(prompt: str) -> str:
    return normalize_body(question_body(prompt))


class Embedder:
    def __init__(self, device: str | None = None) -> None:
        self._device = device

    @cached_property
    def _model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(EMBED_MODEL, device=self._device)

    def encode(self, prompts: list[str]) -> np.ndarray:
        bodies = [routing_text(p) for p in prompts]
        return self._model.encode(bodies, batch_size=128, normalize_embeddings=True, show_progress_bar=False)


class CategoryClassifier:
    """다항 로지스틱 회귀 (가중치는 npz 로 저장해 해시 동결)."""

    def __init__(self, classes: list[str], coef: np.ndarray, intercept: np.ndarray) -> None:
        self.classes, self.coef, self.intercept = list(classes), coef, intercept

    @classmethod
    def fit(cls, X: np.ndarray, y: list[str], C: float = 10.0) -> "CategoryClassifier":
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=C, max_iter=2000).fit(X, y)
        return cls(list(lr.classes_), lr.coef_, lr.intercept_)

    def predict(self, X: np.ndarray) -> list[str]:
        return [self.classes[i] for i in np.argmax(X @ self.coef.T + self.intercept, axis=1)]

    def save(self, path: Path) -> None:
        np.savez(path, coef=self.coef, intercept=self.intercept)
        path.with_suffix(".classes.json").write_text(json.dumps(self.classes))

    @classmethod
    def load(cls, path: Path) -> "CategoryClassifier":
        d = np.load(path)
        return cls(json.loads(path.with_suffix(".classes.json").read_text()), d["coef"], d["intercept"])
