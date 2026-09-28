"""RouterArena와 겹치지 않는 외부 품질 측정 세트를 만든다.

- MMLU-Pro test: 14개 카테고리 x N문항, RouterArena 문항과 텍스트 중복 제거
- AIME 2025: 30문항 (RouterArena AIME와 중복 검사)
출력: data/probe_set.jsonl  {id, source, category, question, options, answer, kind}
"""
import json
import random
import re
import sys

sys.path.insert(0, "scripts")
import _tls  # noqa: F401
from datasets import load_dataset

PER_CATEGORY = 10
SEED = 7


def norm(text: str) -> str:
    return re.sub(r"\W+", " ", text).lower().strip()[:200]


def main() -> None:
    arena = load_dataset("RouteWorks/RouterArena", split="full")
    arena_keys = {norm(q) for q in arena["Question"]}

    rng = random.Random(SEED)
    rows = []

    mmlu = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    by_cat: dict[str, list] = {}
    for r in mmlu:
        by_cat.setdefault(r["category"], []).append(r)
    overlap = 0
    for cat, items in sorted(by_cat.items()):
        rng.shuffle(items)
        picked = 0
        for r in items:
            if norm(r["question"]) in arena_keys:
                overlap += 1
                continue
            rows.append({
                "id": f"mmlupro_{r['question_id']}",
                "source": "MMLU-Pro",
                "category": cat,
                "question": r["question"],
                "options": r["options"],
                "answer": r["answer"],
                "kind": "mcq",
            })
            picked += 1
            if picked == PER_CATEGORY:
                break
    print(f"MMLU-Pro: {sum(1 for r in rows if r['source']=='MMLU-Pro')} picked, {overlap} skipped as RouterArena overlap")

    aime = load_dataset("MathArena/aime_2025", split="train")
    aime_overlap = 0
    for r in aime:
        if norm(r["problem"]) in arena_keys:
            aime_overlap += 1
            continue
        rows.append({
            "id": f"aime2025_{r['problem_idx']}",
            "source": "AIME2025",
            "category": "math",
            "question": r["problem"],
            "options": [],
            "answer": str(r["answer"]),
            "kind": "math",
        })
    print(f"AIME2025: {sum(1 for r in rows if r['source']=='AIME2025')} picked, {aime_overlap} overlap")

    with open("data/probe_set.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"total {len(rows)} -> data/probe_set.jsonl")


if __name__ == "__main__":
    main()
