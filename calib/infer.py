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

from calib.common import DATA_DIR, ROOT

URL = "https://openrouter.ai/api/v1/chat/completions"
KEY_URL = "https://openrouter.ai/api/v1/key"
RESULTS_DIR = DATA_DIR / "results"
RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class SpendCapReached(RuntimeError):
    pass


async def account_usage(client: httpx.AsyncClient, key: str) -> float:
    """계정 누적 사용액. 일시적 네트워크 오류로 전체 실행이 죽지 않도록 재시도한다."""
    for attempt in range(5):
        try:
            r = await client.get(KEY_URL, headers={"Authorization": f"Bearer {key}"})
            return float(r.json()["data"]["usage"])
        except (httpx.HTTPError, KeyError, ValueError):
            if attempt == 4:
                raise
            await asyncio.sleep(3 * 2**attempt)
    raise RuntimeError("unreachable")


def split_variant(model: str) -> tuple[str, dict]:
    """"vendor/model@high" → ("vendor/model", {"reasoning": {"effort": "high"}}), "@off" → 추론 끔. 추론 노력도 라우팅 선택지로 쓴다."""
    if "@" not in model:
        return model, {}
    base, effort = model.split("@", 1)
    if effort == "off":
        return base, {"reasoning": {"enabled": False}}
    if effort.startswith("cap"):  # "@cap1000" → 추론 토큰 상한 (답 자체는 잘리지 않음)
        return base, {"reasoning": {"max_tokens": int(effort[3:])}}
    return base, {"reasoning": {"effort": effort}}


CALL_TIMEOUT_S = 900.0


OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def _openai_direct(base_model: str, extra: dict) -> tuple[str, dict, str] | None:
    """openai/* 모델은 OPENAI_API_KEY 가 있으면 OpenAI API로 직접 호출 (추론 설정은 reasoning_effort 로 변환)."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key or not base_model.startswith("openai/"):
        return None
    body_extra = {}
    r = extra.get("reasoning") or {}
    if r.get("enabled") is False:
        body_extra["reasoning_effort"] = "none"
    elif "effort" in r:
        body_extra["reasoning_effort"] = r["effort"]
    return OPENAI_URL, {"model": base_model.split("/", 1)[1], **body_extra}, key


async def call(client: httpx.AsyncClient, key: str, model: str, prompt: str) -> dict:
    base_model, extra = split_variant(model)
    body = {"model": base_model, "messages": [{"role": "user", "content": prompt}], "usage": {"include": True}, **extra}
    url = URL
    direct = _openai_direct(base_model, extra)
    if direct:
        url, b, key = direct
        body = {**b, "messages": [{"role": "user", "content": prompt}]}
    invoked_at = dt.datetime.now(dt.timezone.utc).isoformat()
    t0 = time.monotonic()
    last_error = ""
    for attempt in range(5):
        try:
            # httpx 타임아웃은 읽기 간격 기준이라 keepalive가 오면 끝나지 않는다 → 요청 전체 시간 상한
            r = await asyncio.wait_for(client.post(url, json=body, headers={"Authorization": f"Bearer {key}"}), CALL_TIMEOUT_S)
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
                "model_used": ("openai/" + data["model"]) if direct and data.get("model") else data.get("model"),
                "provider": data.get("provider") or ("OpenAI" if direct else None),
                "request_id": data.get("id"),
                "request_params": extra,
                "invoked_at": invoked_at,
                "content": choice["message"].get("content"),
                "finish_reason": choice.get("finish_reason"),
                "usage": usage,
                "actual_cost": usage.get("cost"),
                "latency_s": time.monotonic() - t0,
            }
        except (httpx.HTTPError, asyncio.TimeoutError, json.JSONDecodeError, KeyError, IndexError) as e:
            last_error = f"{type(e).__name__}: {e}"[:300]
            await asyncio.sleep(3 * 2**attempt)
    return {"requested_model": model, "invoked_at": invoked_at, "error": last_error, "latency_s": time.monotonic() - t0}


def result_path(model: str, results_dir: Path = RESULTS_DIR) -> Path:
    return results_dir / f"{model.replace('/', '__')}.jsonl"


def done_ids(model: str, results_dir: Path = RESULTS_DIR) -> set[str]:
    path = result_path(model, results_dir)
    if not path.exists():
        return set()
    # 오류 없이 빈 답(공백 포함)이 온 경우도 다시 호출한다
    recs = [json.loads(l) for l in path.read_text().splitlines()]
    return {r["id"] for r in recs if "error" not in r and (r.get("content") or "").strip()}


async def run_pairs(
    pairs: list[tuple[str, dict]], key: str, concurrency: int, spend_cap: float, results_dir: Path = RESULTS_DIR
) -> None:
    """(model, {id, prompt}) 쌍을 호출한다. 이미 성공한 (model, id)는 건너뛴다."""
    results_dir.mkdir(parents=True, exist_ok=True)
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)
    stop = asyncio.Event()
    # 기본 연결 풀(100)이 동시성 상한이 되지 않도록 넉넉히
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0), limits=httpx.Limits(max_connections=1000, max_keepalive_connections=200)) as client:
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




def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--spend-cap", type=float, default=35.0, help="계정 누적 사용액(USD)이 이 값에 도달하면 중단")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
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
    pairs = [(m, it) for it in items for m in models]
    asyncio.run(run_pairs(pairs, key, args.concurrency, args.spend_cap, Path(args.results_dir)))


if __name__ == "__main__":
    main()
