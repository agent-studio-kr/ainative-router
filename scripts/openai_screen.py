"""OpenAI API 직접 호출로 모델·추론 설정 스크리닝 (140문항 MMLU-Pro 비중복 세트).

usage: uv run --native-tls python scripts/openai_screen.py gpt-6-luna:minimal gpt-6-luna:low:vlow ...
설정 문법: model[:reasoning_effort][:vlow] (vlow = verbosity low). 결과는 data/screen_results/openai__<설정>.jsonl
"""
import asyncio, json, os, sys, time
from pathlib import Path
sys.path.insert(0, "scripts"); import _tls  # noqa
import httpx
from dotenv import load_dotenv
load_dotenv(".env"); KEY = os.environ["OPENAI_API_KEY"]
items = [json.loads(l) for l in open("data/probe_mmlupro_prompts.jsonl")]
OUT = Path("data/screen_results")

def body(spec, prompt):
    parts = spec.split(":"); b = {"model": parts[0], "messages": [{"role": "user", "content": prompt}]}
    for p in parts[1:]:
        if p == "vlow": b["verbosity"] = "low"
        else: b["reasoning_effort"] = p
    return b

async def run(spec, client, sem):
    path = OUT / f"openai__{spec.replace(':', '@')}.jsonl"
    done = {json.loads(l)["id"] for l in path.open()} if path.exists() else set()
    async def one(it):
        async with sem:
            t0 = time.monotonic()
            try:
                r = await client.post("https://api.openai.com/v1/chat/completions", json=body(spec, it["prompt"]), headers={"Authorization": f"Bearer {KEY}"})
                d = r.json(); u = d.get("usage") or {}
                rec = {"id": it["id"], "requested_model": spec, "model_used": d.get("model"), "content": d["choices"][0]["message"].get("content"),
                       "usage": {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"), "total_tokens": u.get("total_tokens"),
                                 "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens")}, "latency_s": time.monotonic() - t0}
            except Exception as e:
                rec = {"id": it["id"], "requested_model": spec, "error": f"{type(e).__name__}: {str(d.get('error', e) if 'd' in dir() else e)[:200]}"}
            with path.open("a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    await asyncio.gather(*[one(it) for it in items if it["id"] not in done])

async def main():
    sem = asyncio.Semaphore(100)
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0), limits=httpx.Limits(max_connections=400)) as c:
        await asyncio.gather(*[run(s, c, sem) for s in sys.argv[1:]])
asyncio.run(main())
