"""라우팅 신호: 과제 유형(템플릿 그룹) · 질문 본문 · 도메인.

- 과제 유형: RouterArena 공개 eval config 템플릿의 고정 머리말로 판별 (데이터가 아닌 설정 파일에서 도출).
  머리말이 맞지 않으면(패러프레이즈된 프롬프트 등) 보정 세트 프롬프트 임베딩 kNN으로 폴백한다.
- 도메인: 템플릿을 떼어낸 질문 본문을 Vela-1.0-Encoder-307M-Domain(14 MMLU-Pro 도메인)으로 분류.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

CONFIG_DIR = Path(__file__).resolve().parents[1] / "third_party" / "RouterArena" / "config" / "eval_config" / "zero-shot"
DOMAIN_MODEL = "llm-semantic-router/Vela-1.0-Encoder-307M-Domain"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
KNN_K = 7
KNN_PREFIX_CHARS = 300  # 지시문 위주로 임베딩 (본문 내용이 과제 판별을 지배하지 않도록)


def template_heads(config_dir: Path = CONFIG_DIR) -> dict[str, str]:
    """템플릿 고정 머리말 → 그룹 이름. 같은 머리말을 공유하는 config는 한 그룹 (그룹 이름 = config 이름들을 '|'로 연결)."""
    groups: dict[str, list[str]] = {}
    for p in sorted(config_dir.glob("*.json")):
        params = json.loads(p.read_text())["eval_params"]
        template = params.get("prompt") or params.get("is_stdin_prompt") or ""
        head = template.split("{")[0].strip()
        groups.setdefault(head, []).append(p.stem)
        if "not_is_stdin_prompt" in params:  # LiveCodeBench 함수형 템플릿
            groups.setdefault(params["not_is_stdin_prompt"].split("{")[0].strip(), []).append(p.stem)
    return {head: "|".join(sorted(set(cfgs))) for head, cfgs in groups.items() if head}


_BODY_PATTERNS = [
    re.compile(r"Question:\s*(.*?)\s*(?:\n\s*Options:|\n\s*Provide |\Z)", re.S),
    re.compile(r"Translate the following sentence[^\n]*\n\s*(.*?)\s*(?:\n\s*Provide |\Z)", re.S),
]


def question_body(prompt: str) -> str:
    """템플릿 지시문을 뗀 질문 본문 (도메인 분류용). 패턴이 없으면 원문."""
    for pat in _BODY_PATTERNS:
        m = pat.search(prompt)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return prompt


@dataclass
class Signals:
    task: str  # 템플릿 그룹 이름
    task_source: str  # "template" | "knn"
    domain: str
    domain_conf: float


class SignalExtractor:
    def __init__(self, knn_prompts: list[str], knn_tasks: list[str], device: str | None = None) -> None:
        self.heads = sorted(template_heads().items(), key=lambda kv: -len(kv[0]))  # 긴 머리말 우선
        self._knn_prompts = knn_prompts
        self._knn_tasks = np.array(knn_tasks)
        self._device = device

    @cached_property
    def _embedder(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(EMBED_MODEL, device=self._device)

    @cached_property
    def _knn_matrix(self) -> np.ndarray:
        return self._embedder.encode([p[:KNN_PREFIX_CHARS] for p in self._knn_prompts], batch_size=128, normalize_embeddings=True, show_progress_bar=False)

    @cached_property
    def _domain(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        dev = self._device or ("mps" if torch.backends.mps.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(DOMAIN_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(DOMAIN_MODEL).eval().to(dev)
        return tok, model, dev

    def task_by_template(self, prompt: str) -> str | None:
        p = prompt.lstrip()
        for head, group in self.heads:
            if p.startswith(head):
                return group
        return None

    def task_by_knn(self, prompts: list[str]) -> list[str]:
        q = self._embedder.encode([p[:KNN_PREFIX_CHARS] for p in prompts], batch_size=128, normalize_embeddings=True, show_progress_bar=False)
        sims = q @ self._knn_matrix.T
        top = np.argsort(-sims, axis=1)[:, :KNN_K]
        out = []
        for row in top:
            labels, counts = np.unique(self._knn_tasks[row], return_counts=True)
            out.append(str(labels[np.argmax(counts)]))
        return out

    def domains(self, bodies: list[str], batch_size: int = 32) -> list[tuple[str, float]]:
        import torch

        tok, model, dev = self._domain
        out: list[tuple[str, float]] = []
        with torch.no_grad():
            for i in range(0, len(bodies), batch_size):
                enc = tok(bodies[i : i + batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
                probs = model(**enc).logits.softmax(-1).cpu().numpy()
                out += [(model.config.id2label[int(p.argmax())], float(p.max())) for p in probs]
        return out

    def extract(self, prompts: list[str]) -> list[Signals]:
        tasks = [self.task_by_template(p) for p in prompts]
        miss = [i for i, t in enumerate(tasks) if t is None]
        if miss:
            for i, t in zip(miss, self.task_by_knn([prompts[i] for i in miss])):
                tasks[i] = t
        doms = self.domains([question_body(p) for p in prompts])
        miss_set = set(miss)
        return [
            Signals(task=t, task_source="knn" if i in miss_set else "template", domain=d, domain_conf=c)
            for i, (t, (d, c)) in enumerate(zip(tasks, doms))
        ]
