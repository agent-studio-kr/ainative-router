"""템플릿 머리말 패러프레이즈 생성 (kNN 폴백 테스트용, RouterArena 데이터 미사용).

RouterArena robustness split은 지시문을 패러프레이즈·동의어·문법 변경·오타로 섭동한다(논문).
공개 config 템플릿 머리말마다 같은 유형의 변형 3개를 LLM으로 만든다.
출력: data/calib/template_paraphrases.json  {group: [head, para1, para2, para3]}
"""
import asyncio
import json
import os
import re
import sys

import httpx
from dotenv import load_dotenv

sys.path.insert(0, ".")
from arena_router.signals import template_heads  # noqa: E402
from calib.common import DATA_DIR, ROOT  # noqa: E402
from calib.infer import call  # noqa: E402

PROMPT = (
    "Rewrite the following instruction in 3 different ways: (1) a paraphrase with different wording, "
    "(2) synonym substitutions with slightly changed grammar, (3) a paraphrase that also contains 1-2 typos. "
    "Keep the meaning. Return ONLY a JSON list of 3 strings.\n\nInstruction: {head}"
)


async def main() -> None:
    load_dotenv(ROOT / ".env")
    key = os.environ["OPENROUTER_API_KEY"]
    heads = template_heads()
    out = {}
    async with httpx.AsyncClient(timeout=120) as c:
        res = await asyncio.gather(*(call(c, key, "openai/gpt-6-luna", PROMPT.format(head=h)) for h in heads))
    for (head, group), r in zip(heads.items(), res):
        m = re.search(r"\[.*\]", r.get("content") or "", re.S)
        paras = json.loads(m.group(0)) if m else []
        out.setdefault(group, []).append({"head": head, "paraphrases": paras[:3]})
    (DATA_DIR / "template_paraphrases.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(sum(len(v) for v in out.values()), "heads paraphrased")


if __name__ == "__main__":
    asyncio.run(main())
