"""OpenRouter 추론 클라이언트 (출처 보존 + 지출 하드 스톱).

입력 jsonl: {"id", "prompt", ...}  → 출력: data/calib/results/<model>.jsonl (id 단위 캐시, 재실행 시 이어서)
각 행에 requested_model / model_used / provider / invoked_at / request_id / usage / actual_cost 를 기록한다.
공식 비용(토큰 × 가격표)은 채점 단계에서 계산한다 — 여기의 actual_cost 는 공급자 청구액.

usage:
  uv run --native-tls python -m calib.infer --input data/calib/calib_prompts.jsonl --models a/b,c/d [--limit N] [--spend-cap 35]
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

from calib.common import ROOT

URL = "https://openrouter.ai/api/v1/chat/completions"
KEY_URL = "https://openrouter.ai/api/v1/key"
RESULTS_DIR = ROOT / "data" / "calib" / "results"
RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class SpendCapReached(RuntimeError):
    pass


async def account_usage(client: httpx.AsyncClient, key: str) -> float:
    r = await client.get(KEY_URL, headers={"Authorization": f"Bearer {key}"})
    return float(r.json()["data"]["usage"])


async def call(client: httpx.AsyncClient, key: str, model: str, prompt: str) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "usage": {"include": True}}
    invoked_at = dt.datetime.now(dt.timezone.utc).isoformat()
    t0 = time.monotonic()
    last_error = ""
    for attempt in range(5):
        try:
            r = await client.post(URL, json=body, headers={"Authorization": f"Bearer {key}"})
            data = r.json()
            if r.status_code in RETRY_STATUS or ("error" in data and data["error"].get("code") in RETRY_STATUS):
                last_error = str(data.get("error", r.status_code))[:300]
                await asyncio.sleep(3 * 2**attempt)
                continue
            if "error" in data:
                last_error = str(data["error"])[:300]
                if attempt < 2:  # 공급자 일시 오류(400 Provider returned error 등)도 2회까지 재시도
                    await asyncio.sleep(3 * 2**attempt)
                    continue
                break
            choice = data["choices"][0]
            usage = data.get("usage") or {}
            return {
                "requested_model": model,
                "model_used": data.get("model"),
                "provider": data.get("provider"),
                "request_id": data.get("id"),
                "invoked_at": invoked_at,
                "content": choice["message"].get("content"),
                "finish_reason": choice.get("finish_reason"),
                "usage": usage,
                "actual_cost": usage.get("cost"),
                "latency_s": time.monotonic() - t0,
            }
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError) as e:
            last_error = f"{type(e).__name__}: {e}"[:300]
            await asyncio.sleep(3 * 2**attempt)
    return {"requested_model": model, "invoked_at": invoked_at, "error": last_error, "latency_s": time.monotonic() - t0}


def result_path(model: str, results_dir: Path = RESULTS_DIR) -> Path:
    return results_dir / f"{model.replace('/', '__')}.jsonl"


def done_ids(model: str, results_dir: Path = RESULTS_DIR) -> set[str]:
    path = result_path(model, results_dir)
    if not path.exists():
        return set()
    return {json.loads(l)["id"] for l in path.read_text().splitlines() if "error" not in json.loads(l)}


async def run_pairs(
    pairs: list[tuple[str, dict]], key: str, concurrency: int, spend_cap: float, results_dir: Path = RESULTS_DIR
) -> None:
    """(model, {id, prompt}) 쌍을 호출한다. 이미 성공한 (model, id)는 건너뛴다."""
    results_dir.mkdir(parents=True, exist_ok=True)
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)
    stop = asyncio.Event()
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        start_usage = await account_usage(client, key)
        print(f"account usage at start: ${start_usage:.2f} (cap ${spend_cap:.2f})", flush=True)
        counter = {"n": 0}

        async def guard() -> None:
            counter["n"] += 1
            if counter["n"] % 100 == 0:
                used = await account_usage(client, key)
                print(f"  {counter['n']} calls, account usage ${used:.2f}", flush=True)
                if used >= spend_cap:
                    stop.set()

        async def one(model: str, item: dict) -> None:
            if stop.is_set():
                return
            async with sem:
                if stop.is_set():
                    return
                res = await call(client, key, model, item["prompt"])
            rec = {"id": item["id"], **res}
            async with lock:
                with result_path(model, results_dir).open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            await guard()

        skip = {m: done_ids(m, results_dir) for m in {m for m, _ in pairs}}
        tasks = [one(m, it) for m, it in pairs if it["id"] not in skip[m]]
        print(f"{len(tasks)} calls to make", flush=True)
        await asyncio.gather(*tasks)
        end_usage = await account_usage(client, key)
        print(f"spent this run: ${end_usage - start_usage:.2f} (account ${end_usage:.2f})", flush=True)
        if stop.is_set():
            raise SpendCapReached(f"spend cap ${spend_cap} reached")


async def run(items: list[dict], models: list[str], key: str, concurrency: int, spend_cap: float) -> None:
    # 모델을 번갈아 배치해 한 모델이 느려도 전체가 막히지 않게 한다
    pairs = [(m, it) for it in items for m in models]
    await run_pairs(pairs, key, concurrency, spend_cap)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--spend-cap", type=float, default=35.0, help="계정 누적 사용액(USD)이 이 값에 도달하면 중단")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    key = os.environ["OPENROUTER_API_KEY"]
    items = [json.loads(l) for l in Path(args.input).read_text().splitlines()]
    if args.ids:
        wanted = set(args.ids.split(","))
        items = [it for it in items if it["id"] in wanted]
    if args.limit:
        items = items[: args.limit]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    asyncio.run(run(items, models, key, args.concurrency, args.spend_cap))


if __name__ == "__main__":
    main()
