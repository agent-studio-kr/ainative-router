"""M0: 추론 결과 행에 출처 필드가 모두 보존되는지 실제 1건 호출로 검증 (~$0.0001)."""
import asyncio
import os

import httpx
import pytest
from dotenv import load_dotenv

from calib.common import ROOT
from calib.infer import call

REQUIRED = ["requested_model", "model_used", "provider", "request_id", "invoked_at", "content", "usage", "actual_cost"]


@pytest.mark.skipif(not (ROOT / ".env").exists(), reason="no API key")
def test_call_preserves_provenance():
    load_dotenv(ROOT / ".env")

    async def go():
        async with httpx.AsyncClient(timeout=120) as c:
            return await call(c, os.environ["OPENROUTER_API_KEY"], "google/gemma-4-31b-it", "Reply with \\boxed{A}.")

    rec = asyncio.run(go())
    assert "error" not in rec, rec
    missing = [k for k in REQUIRED if rec.get(k) in (None, "")]
    assert not missing, missing
    assert rec["usage"].get("completion_tokens", 0) > 0
