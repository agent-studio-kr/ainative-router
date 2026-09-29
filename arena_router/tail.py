"""추론 길이 예측기: 질문 본문만으로 "기본 모델이 오래 생각할 문항"(비용 꼬리)을 예측한다.

입력 = routing_text(본문, 형식 단서 제거)의 단어 1~2-gram TF-IDF + 본문 임베딩 + 계산형 특징(숫자·단위·보기의 숫자 비율 등).
목표 = 외부 보정 문항에서 관측한 deepseek-v4.1-flash 추론 토큰(low/기본/cap1000 세 실행 평균)의 log1p. Ridge 회귀.
가중치는 npz/json 으로 저장해 해시 동결한다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from arena_router.signals import question_body, routing_text

_OPT_LINE = re.compile(r"^\s*\(?[A-Ja-j][\).:\]]\s+")
_UNIT = re.compile(r"(?i)\b(kg|km|m/s|mol|joule|volt|ohm|watt|kpa|psi|btu|hz|°|%|\$)")
_CALC = re.compile(r"(?i)calculat|compute|determine|estimate|find the|how many|how much|what is the (value|rate|probability|magnitude)")


def numeric_features(prompt: str) -> list[float]:
    text, body = routing_text(prompt), question_body(prompt)
    opts = [l for l in body.split("\n") if l.strip() and _OPT_LINE.match(l)]
    return [
        float(np.log1p(len(text))),
        float(np.log1p(len(re.findall(r"\d+(?:\.\d+)?", text)))),
        sum(bool(re.search(r"\d", l)) for l in opts) / max(1, len(opts)),
        float(len(opts)),
        float(bool(_UNIT.search(text))),
        float(bool(_CALC.search(text))),
        sum(c in "=+-*/^√" for c in text) / max(1, len(text)),
        float(np.mean([len(l) for l in opts])) if opts else 0.0,
    ]


class TailPredictor:
    def __init__(self, vocab: dict[str, int], idf: np.ndarray, mean: np.ndarray, scale: np.ndarray, coef: np.ndarray, intercept: float) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vocab, self.idf, self.mean, self.scale, self.coef, self.intercept = vocab, idf, mean, scale, coef, intercept
        self._vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, vocabulary=vocab)
        self._vec.idf_ = idf

    def _features(self, prompts: list[str], X_emb: np.ndarray):
        from scipy.sparse import csr_matrix, hstack

        N = (np.array([numeric_features(p) for p in prompts]) - self.mean) / self.scale
        return hstack([self._vec.transform([routing_text(p) for p in prompts]), csr_matrix(N), csr_matrix(X_emb)]).tocsr()

    def predict(self, prompts: list[str], X_emb: np.ndarray) -> np.ndarray:
        return self._features(prompts, X_emb) @ self.coef + self.intercept

    @classmethod
    def fit(cls, prompts: list[str], X_emb: np.ndarray, target: np.ndarray, alpha: float = 3.0) -> "TailPredictor":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import Ridge

        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000).fit([routing_text(p) for p in prompts])
        N = np.array([numeric_features(p) for p in prompts])
        mean, scale = N.mean(0), N.std(0)
        scale[scale == 0] = 1.0
        vocab = {t: int(i) for t, i in v.vocabulary_.items()}
        tp = cls(vocab, v.idf_, mean, scale, np.zeros(0), 0.0)
        r = Ridge(alpha=alpha).fit(tp._features(prompts, X_emb), target)
        tp.coef, tp.intercept = r.coef_, float(r.intercept_)
        return tp

    def save(self, path: Path) -> None:
        np.savez(path, idf=self.idf, mean=self.mean, scale=self.scale, coef=self.coef, intercept=np.array([self.intercept]))
        path.with_suffix(".vocab.json").write_text(json.dumps(self.vocab, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> "TailPredictor":
        d = np.load(path)
        vocab = json.loads(path.with_suffix(".vocab.json").read_text())
        return cls(vocab, d["idf"], d["mean"], d["scale"], d["coef"], float(d["intercept"][0]))
