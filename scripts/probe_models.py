"""OpenRouter 후보 모델의 품질·실비용을 외부 측정 세트로 잰다.

RouterArena의 OpenRouter 호출과 동일 조건(기본 파라미터, 동일 zero-shot 템플릿)을 쓰고,
실제 과금액은 OpenRouter usage.cost로 기록한다. 결과는 (model, id) 단위로 캐시되어 재실행 시 이어서 진행.

usage:
  uv run --native-tls python scripts/probe_models.py --models a/b,c/d [--limit N] [--concurrency 16]
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "scripts")
import _tls  # noqa: F401
import httpx
from dotenv import load_dotenv

PROBE_SET = Path("data/probe_set.jsonl")
RESULTS_DIR = Path("data/probe_results")
URL = "https://openrouter.ai/api/v1/chat/completions"

# RouterArena config/eval_config/zero-shot/{MMLUPro,AIME}.json 템플릿 (포맷 후 형태)
MCQ_TEMPLATE = (
    "Please read the following multiple-choice questions and provide the most likely correct answer "
    "based on the options given.\n\nContext: None\n\nQuestion: {question}\n\nOptions: \n{options}\n\n"
    "Provide the correct letter choice in \\boxed{{X}}, where X is the correct letter choice. "
    "Keep the explanation or feedback within 3 sentences."
)
MATH_TEMPLATE = (
    "Please solve the following mathematical problem step by step. \n\nContext: \n\nQuestion: {question}\n\n"
    "Provide your final answer in \\boxed{{}} format, where the content inside the braces is the exact "
    "mathematical expression or number. For example: \\boxed{{42}}. "
    "Keep your explanation clear, concise, and within 3 sentences."
)


def build_prompt(item: dict) -> str:
    if item["kind"] == "mcq":
        options = "".join(f"{chr(65 + i)}. {opt}\n" for i, opt in enumerate(item["options"]))
        return MCQ_TEMPLATE.format(question=item["question"], options=options)
    return MATH_TEMPLATE.format(question=item["question"])


def last_boxed(text: str) -> str | None:
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth, i = 0, start + len("\\boxed")
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j].strip()
    return None


def score(item: dict, content: str | None) -> float:
    if not content:
        return 0.0
    boxed = last_boxed(content)
    if item["kind"] == "mcq":
        cand = boxed or ""
        m = re.findall(r"\b([A-J])\b", cand.upper())
        return 1.0 if m and m[-1] == item["answer"] else 0.0
    if boxed is None:
        return 0.0
    digits = re.sub(r"[^\d-]", "", boxed.replace("\\,", ""))
    try:
        return 1.0 if int(digits) == int(item["answer"]) else 0.0
    except ValueError:
        return 0.0


async def call(client: httpx.AsyncClient, key: str, model: str, prompt: str) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "usage": {"include": True}}
    t0 = time.monotonic()
    for attempt in range(4):
        try:
            r = await client.post(URL, json=body, headers={"Authorization": f"Bearer {key}"})
            if r.status_code in (429, 500, 502, 503) and attempt < 3:
                await asyncio.sleep(2 ** attempt * 3)
                continue
            data = r.json()
            if "error" in data:
                return {"error": str(data["error"])[:300], "latency_s": time.monotonic() - t0}
            choice = data["choices"][0]
            usage = data.get("usage") or {}
            return {
                "content": choice["message"].get("content"),
                "finish_reason": choice.get("finish_reason"),
                "provider": data.get("provider"),
                "usage": usage,
                "cost": usage.get("cost"),
                "latency_s": time.monotonic() - t0,
            }
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError) as e:
            if attempt == 3:
                return {"error": f"{type(e).__name__}: {e}"[:300], "latency_s": time.monotonic() - t0}
            await asyncio.sleep(2 ** attempt * 3)
    return {"error": "exhausted retries"}


async def run_model(model: str, items: list[dict], key: str, concurrency: int) -> None:
    out_path = RESULTS_DIR / f"{model.replace('/', '__')}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            rec = json.loads(line)
            if "error" not in rec:
                done.add(rec["id"])
    todo = [it for it in items if it["id"] not in done]
    if not todo:
        return
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:

        async def one(item: dict) -> None:
            async with sem:
                res = await call(client, key, model, build_prompt(item))
            rec = {"id": item["id"], "source": item["source"], "category": item["category"], "model": model, **res}
            rec["accuracy"] = score(item, res.get("content"))
            async with lock:
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        await asyncio.gather(*(one(it) for it in todo))
    print(f"[done] {model}: {len(todo)} calls", flush=True)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--limit", type=int, default=0, help="앞에서부터 N문항만 (스모크 테스트용)")
    ap.add_argument("--ids", default="", help="특정 id만 (쉼표 구분)")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    load_dotenv()
    key = os.environ["OPENROUTER_API_KEY"]
    items = [json.loads(l) for l in PROBE_SET.read_text().splitlines()]
    if args.ids:
        wanted = set(args.ids.split(","))
        items = [it for it in items if it["id"] in wanted]
    if args.limit:
        items = items[: args.limit]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    await asyncio.gather(*(run_model(m, items, key, args.concurrency) for m in models))


if __name__ == "__main__":
    asyncio.run(main())
