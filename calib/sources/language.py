"""언어 계열 보정 원천 로더 (WMT19 · NarrativeQA · SuperGLUE · QANTA · GeoGraphyData).

RouterArena 가 쓴 원본 HF 원천에서 RouterArena 문항을 제외하고 같은 표현(dataset_name·question/context·options·answer)으로
CalibRow 를 만든다. 원천·표현 근거·대체 원천은 data/calib/sources_language.md 에 기록한다.

    LOADERS[config_name](target, seed, excl) -> list[CalibRow]   # 최대 ceil(1.3 × target) 개
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from typing import Callable

from calib.common import ROOT, CalibRow, ExclusionIndex, normalize

OVERSAMPLE = 1.3
TARGETS = json.loads((ROOT / "calib" / "targets.json").read_text())
WMT_PAIRS = ["cs", "de", "fi", "gu", "kk", "lt", "ru", "zh"]


# ---------------------------------------------------------------- 공통 헬퍼
def _n(target: int) -> int:
    return math.ceil(target * OVERSAMPLE)


def _rng(seed: int, name: str) -> random.Random:
    return random.Random(f"{seed}:{name}")


def _h(text: str) -> str:
    return hashlib.sha1(normalize(text).encode()).hexdigest()[:12]


def _spread(rows: list[CalibRow], n: int, rng: random.Random) -> list[CalibRow]:
    """결정적 셔플 후 그룹(문서/지문)별로 돌아가며 뽑는다 — 같은 그룹이 몰리지 않게."""
    rows = sorted(rows, key=lambda r: r.source_id)
    rng.shuffle(rows)
    rank: dict[str, int] = defaultdict(int)
    keyed = []
    for i, r in enumerate(rows):
        keyed.append((rank[r.group_id], i, r))
        rank[r.group_id] += 1
    return [r for _, _, r in sorted(keyed, key=lambda t: t[:2])][:n]


def _allocate(n: int, weights: dict[str, int]) -> dict[str, int]:
    """최대 잉여 방식으로 n 을 weights 비율로 나눈다."""
    total = sum(weights.values())
    raw = {k: n * w / total for k, w in weights.items()}
    quota = {k: math.floor(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (quota[k] - raw[k], k))[: n - sum(quota.values())]:
        quota[k] += 1
    return quota


def _keep(rows: list[CalibRow], excl: ExclusionIndex, key: str) -> list[CalibRow]:
    """RouterArena 문항(질문/컨텍스트 일치) + 원천 ID 단위 제외 목록(excl.source_ids[key])을 걸러낸다."""
    ids = excl.source_ids.get(key, set())
    return [r for r in rows if r.source_id not in ids and r.group_id not in ids and not excl.is_excluded(r.question, r.context)]


def _ra_questions(excl: ExclusionIndex, prefix: str) -> list[str]:
    return [q for name, qs in excl.questions_by_dataset.items() if name.startswith(prefix) for q in qs]


def _load(repo: str, config: str | None, split: str):
    from datasets import load_dataset

    return load_dataset(repo, config, split=split)


def _float_label(label: int) -> str:
    return str(float(label))  # RouterArena 는 0/1 라벨을 "0.0"/"1.0" 문자열로 저장


def _yes_no(label: int) -> str:
    return "Yes" if label == 1 else "No"


# ---------------------------------------------------------------- WMT19
def _wmt_loader(pair: str) -> Callable[[int, int, ExclusionIndex], list[CalibRow]]:
    name = f"WMT19-{pair}-en"

    def load(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
        # RouterArena 는 wmt/wmt19 validation(newstest2018)에서 뽑았고, Question=영어 · Answer=상대 언어 (프롬프트가 "English to X").
        # HF wmt19 에는 train(대용량 웹 코퍼스)/validation 뿐이라 같은 validation 에서 RouterArena 문장쌍을 제외하고 쓴다.
        ds = _load("wmt/wmt19", f"{pair}-en", "validation")
        used = {normalize(q) for q in excl.questions_by_dataset.get(name, [])}
        excl.source_ids[name] = used_ids = set()
        rows = []
        for i, t in enumerate(ds["translation"]):
            sid = f"validation:{i}"
            if normalize(t["en"]) in used:
                used_ids.add(sid)
            rows.append(
                CalibRow(
                    dataset_name=name,
                    config_name=name,
                    question=t["en"],
                    answer=t[pair],
                    source_hf=f"wmt/wmt19:{pair}-en:validation",
                    source_id=sid,
                    group_id=f"wmt19-{pair}:{sid}",
                )
            )
        rows = [r for r in _keep(rows, excl, name) if r.question.strip() and r.answer.strip()]
        return _spread(rows, _n(target), _rng(seed, name))

    return load


# ---------------------------------------------------------------- NarrativeQA
def load_narrativeqa(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    """RouterArena: deepmind/narrativeqa test, Context=document.summary.text(원문 그대로, 앞 공백 포함), Answer=answers[0].text.
    여기서는 validation(문서 단위로 test 와 분리된 split)을 쓰고, RouterArena 요약문과 일치하는 문서는 통째로 제외한다."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    ra_docs = {_h(r) for r in _ra_contexts(excl, "NarrativeQA")}
    rows = []
    for k in range(3):
        path = hf_hub_download("deepmind/narrativeqa", f"data/validation-0000{k}-of-00003.parquet", repo_type="dataset")
        cols = ["document.id", "document.summary.text", "question.text", "answers"]
        table = pq.read_table(path, columns=cols)  # 중첩 필드만 읽으면 컬럼명이 잎 이름이 되므로 위치로 접근 (본문 text 는 읽지 않음)
        for doc_id, summary, question, answers in zip(*(table.column(i).to_pylist() for i in range(4))):
            if _h(summary) in ra_docs:
                excl.source_ids.setdefault("NarrativeQA", set()).add(doc_id)
            rows.append(
                CalibRow(
                    dataset_name="NarrativeQA",
                    config_name="NarrativeQA",
                    question=question,
                    answer=answers[0]["text"],
                    context=summary,
                    source_hf="deepmind/narrativeqa::validation",
                    source_id=f"{doc_id}:{_h(question)}",
                    group_id=doc_id,
                )
            )
    return _spread(_keep(rows, excl, "NarrativeQA"), _n(target), _rng(seed, "NarrativeQA"))


def _ra_contexts(excl: ExclusionIndex, dataset_name: str) -> list[str]:
    # ExclusionIndex 는 컨텍스트를 접두 300자로만 보관하므로 원문 컨텍스트는 RouterArena 에서 다시 읽는다 (제외 목록 용도 한정).
    from datasets import load_dataset

    full = load_dataset("RouteWorks/RouterArena", split="full")
    return [r["Context"] for r in full if r["Dataset name"] == dataset_name and r["Context"]]


# ---------------------------------------------------------------- SuperGLUE
# RouterArena Global Index 는 aps/super_glue validation(axb/axg 는 test)을 config 알파벳순으로 이어 붙인 번호와 일치한다:
# axb 0-1103, boolq 1104-4373, cb 4374-4429, copa 4430-4529, multirc 4530-9377, record 9378-19377, rte 19378-19654, wic, wsc.
# 여기서는 train split 을 쓰고(axb 는 test 뿐이라 예외), 원천 지문 단위로 RouterArena 와 겹치는 지문을 제외한다.
def _sg_row(config: str, sg: str, split: str, sid: str, group: str, **kw) -> CalibRow:
    return CalibRow(dataset_name=config, config_name=config, source_hf=f"aps/super_glue:{sg}:{split}", source_id=sid, group_id=group, **kw)


def load_copa(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    name = "SuperGLUE-CausalReasoning"
    # 전제문이 짧아(50자 미만) ExclusionIndex 컨텍스트 제외가 작동하지 않으므로 RouterArena 전제문 단위로 직접 제외
    excl.source_ids[name] = {_h(c) for c in _ra_contexts(excl, name)}
    rows = []
    for x in _load("aps/super_glue", "copa", "train"):
        q = "what's the effect of this?" if x["question"] == "effect" else "what's the cause for this?"
        rows.append(
            _sg_row(name, "copa", "train", f"copa:train:{x['idx']}", _h(x["premise"]),
                    question=q, context=x["premise"], options=[x["choice1"], x["choice2"]], answer=_float_label(x["label"]))
        )
    # COPA 의 Question 은 고정 템플릿 2종이라 질문 일치 제외(is_excluded)를 쓰면 전부 걸린다 → 전제문(Context) 단위로만 제외.
    rows = [r for r in rows if r.group_id not in excl.source_ids[name] and not excl.is_excluded(r.context)]
    return _spread(rows, _n(target), _rng(seed, name))


def load_record(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    name = "SuperGLUE-ClozeTest"
    ra_passages = {_h(q.split("\nQuery: ")[0]) for q in excl.questions_by_dataset.get(name, [])}
    excl.source_ids[name] = ra_passages
    rows = []
    for x in _load("aps/super_glue", "record", "train"):
        rows.append(
            _sg_row(name, "record", "train", f"record:train:{x['idx']['passage']}:{x['idx']['query']}",
                    _h(f"Passage: {x['passage']}"),  # 지문 해시 = RouterArena 지문 제외 키
                    question=f"Passage: {x['passage']}\nQuery: {x['query']}", options=list(x["entities"]),
                    answer=", ".join(x["answers"]))
        )
    return _spread(_keep(rows, excl, name), _n(target), _rng(seed, name))


def _pair(a: str, b: str = "") -> str:
    return "pair:" + ":".join(sorted([_h(a), _h(b)]))


def load_entailment(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    """RouterArena SuperGLUE-Entailment = RTE validation 36 + AX-b(test, 라벨 공개) 30. 같은 비율로 RTE train + AX-b 비RouterArena 문항."""
    name = "SuperGLUE-Entailment"
    # AX-b 는 같은 문장쌍을 전제/가설만 뒤바꾼 항목이 많다 → 순서 무관 문장쌍 키로 RouterArena 문항과 겹치는 쌍을 제외
    excl.source_ids[name] = {_pair(*q[len("Premise: "):].split(", Hypothesis: ", 1)) for q in excl.questions_by_dataset.get(name, [])}
    rte = [
        _sg_row(name, "rte", "train", f"rte:train:{x['idx']}", _pair(x["premise"], x["hypothesis"]),
                question=f"Premise: {x['premise']}, Hypothesis: {x['hypothesis']}", answer=_float_label(x["label"]))
        for x in _load("aps/super_glue", "rte", "train")
    ]
    axb = [
        _sg_row(name, "axb", "test", f"axb:test:{x['idx']}", _pair(x["sentence1"], x["sentence2"]),
                question=f"Premise: {x['sentence1']}, Hypothesis: {x['sentence2']}", answer=_float_label(x["label"]))
        for x in _load("aps/super_glue", "axb", "test")
    ]
    quota = _allocate(_n(target), {"rte": 36, "axb": 30})
    rng = _rng(seed, name)
    return _spread(_keep(rte, excl, name), quota["rte"], rng) + _spread(_keep(axb, excl, name), quota["axb"], rng)


def load_boolq(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    name = "SuperGLUE-QA"
    rows = []
    for i, x in enumerate(_load("aps/super_glue", "boolq", "train")):
        rows.append(
            _sg_row(name, "boolq", "train", f"boolq:train:{i}", f"boolq:{_h(x['passage'].split(' -- ')[0])}",
                    question=x["question"], context=x["passage"], answer=_float_label(x["label"]))
        )
    return _spread(_keep(rows, excl, name), _n(target), _rng(seed, name))


def load_multirc(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    name = "SuperGLUE-RC"
    ra_paragraphs = {_h(q.split("\nQuestion: ")[0]) for q in excl.questions_by_dataset.get(name, [])}
    excl.source_ids[name] = ra_paragraphs
    rows = []
    for x in _load("aps/super_glue", "multirc", "train"):
        para = f"Paragraph: {x['paragraph']}"
        idx = x["idx"]
        rows.append(
            _sg_row(name, "multirc", "train", f"multirc:train:{idx['paragraph']}:{idx['question']}:{idx['answer']}", _h(para),
                    question=f"{para}\nQuestion: {x['question']}", options=[x["answer"]], answer=_float_label(x["label"]))
        )
    return _spread(_keep(rows, excl, name), _n(target), _rng(seed, name))


def load_wic(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    name = "SuperGLUE-Wic"
    rows = [
        _sg_row(name, "wic", "train", f"wic:train:{x['idx']}", f"wic:{x['word']}",
                question=f"Sentence 1: {x['sentence1']}\nSentence 2: {x['sentence2']}\n", context=x["word"],
                options=["Yes", "No"], answer=_yes_no(x["label"]))
        for x in _load("aps/super_glue", "wic", "train")
    ]
    # Wic 는 Context 가 단어 하나라 context 제외가 작동하지 않는다 → 두 문장 조합(질문) 일치로만 제외.
    return _spread(_keep(rows, excl, name), _n(target), _rng(seed, name))


def load_wsc(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    name = "SuperGLUE-Wsc"
    ra_texts = {_h(q.split("\n span2_text: ")[0]) for q in excl.questions_by_dataset.get(name, [])}
    excl.source_ids[name] = ra_texts
    rows = [
        _sg_row(name, "wsc", "train", f"wsc:train:{x['idx']}", _h(f"Text: {x['text']}"),
                question=f"Text: {x['text']}\n span2_text: {x['span2_text']}\n span1_text: {x['span1_text']}\n",
                options=["Yes", "No"], answer=_yes_no(x["label"]))
        for x in _load("aps/super_glue", "wsc", "train")  # RouterArena 는 wsc.fixed 가 아닌 wsc config 와 일치
    ]
    return _spread(_keep(rows, excl, name), _n(target), _rng(seed, name))


# ---------------------------------------------------------------- QANTA
QANTA_SUBSETS = {k.split("|", 1)[1]: v for k, v in TARGETS["routerarena_subsets"].items() if k.startswith("QANTA|")}


def load_qanta(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    """RouterArena: community-datasets/qanta "mode=first" guesstest(+buzztest/adversarial 일부), Question=first_sentence,
    Answer=page(위키 제목, 밑줄), Dataset name="QANTA_<category>". 여기서는 guessdev+buzzdev 를 쓰고 RouterArena qanta_id 를 제외한다."""
    from datasets import load_dataset

    ds = load_dataset("community-datasets/qanta", "mode=first,char_skip=25")
    ra_q = {normalize(q) for q in _ra_questions(excl, "QANTA_")}
    ra_ids = excl.source_ids.setdefault("QANTA", set())
    for split in ("guesstest", "buzztest", "adversarial"):
        for qid, text in zip(ds[split]["qanta_id"], ds[split]["first_sentence"]):
            if normalize(text) in ra_q:
                ra_ids.add(str(qid))
    rows, seen = [], set()
    for split in ("guessdev", "buzzdev"):
        for x in ds[split]:
            sub = f"QANTA_{x['category']}"
            if sub not in QANTA_SUBSETS or x["qanta_id"] in seen or not x["page"]:
                continue
            seen.add(x["qanta_id"])
            rows.append(
                CalibRow(
                    dataset_name=sub,
                    config_name="QANTA",
                    question=x["first_sentence"],
                    answer=x["page"],
                    source_hf=f"community-datasets/qanta:mode=first,char_skip=25:{split}",
                    source_id=str(x["qanta_id"]),
                    group_id=f"qanta:{x['qanta_id']}",
                )
            )
    rows = _keep(rows, excl, "QANTA")
    by_sub: dict[str, list[CalibRow]] = defaultdict(list)
    for r in rows:
        by_sub[r.dataset_name].append(r)
    rng = _rng(seed, "QANTA")
    out: list[CalibRow] = []
    for sub, k in sorted(_allocate(_n(target), QANTA_SUBSETS).items()):
        out += _spread(by_sub[sub], k, rng)
    return out


# ---------------------------------------------------------------- GeoGraphyData (대체 원천)
GEO_RE = re.compile(
    r"\b(capital|country|countries|city|cities|river|mountain|ocean|sea|lake|island|desert|continent|volcano|"
    r"peninsula|border|borders|province|canal|strait|waterfall|mountains|rivers|islands|lakes)\b",
    re.I,
)


def _geo_answer(a: dict) -> str:
    # TriviaQA value 는 "MAINE" 처럼 대문자인 경우가 있어, 정규화 값이 같으면 위키 개체명 표기(예: "Maine")를 쓴다.
    wiki = a.get("matched_wiki_entity_name") or ""
    return wiki if wiki and normalize(wiki) == normalize(a["value"]) else a["value"]


def load_geography(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    """GeoGraphyData_100k 원본 HF 원천을 찾지 못해(HF/웹/GitHub 검색) 같은 형식(짧은 지리 사실 질문 → 짧은 개체명 정답, exact_match)의
    실데이터 대체 원천을 쓴다: mandarjoshi/trivia_qa rc.nocontext validation 중 지리 키워드 질문 + 4단어 이하 위키 개체 정답."""
    name = "GeoGraphyData_100k"
    rows = []
    for x in _load("mandarjoshi/trivia_qa", "rc.nocontext", "validation"):
        a = x["answer"]
        if a["type"] != "WikipediaEntity" or len(a["value"].split()) > 4 or not GEO_RE.search(x["question"]):
            continue
        rows.append(
            CalibRow(
                dataset_name=name,
                config_name="GeoGraphyData",
                question=x["question"],
                answer=_geo_answer(a),
                source_hf="mandarjoshi/trivia_qa:rc.nocontext:validation",
                source_id=x["question_id"],
                group_id=f"triviaqa:{x['question_id']}",
            )
        )
    return _spread(_keep(rows, excl, name), _n(target), _rng(seed, name))


LOADERS: dict[str, Callable[[int, int, ExclusionIndex], list[CalibRow]]] = {
    **{f"WMT19-{p}-en": _wmt_loader(p) for p in WMT_PAIRS},
    "NarrativeQA": load_narrativeqa,
    "SuperGLUE-CausalReasoning": load_copa,
    "SuperGLUE-ClozeTest": load_record,
    "SuperGLUE-Entailment": load_entailment,
    "SuperGLUE-QA": load_boolq,
    "SuperGLUE-RC": load_multirc,
    "SuperGLUE-Wic": load_wic,
    "SuperGLUE-Wsc": load_wsc,
    "QANTA": load_qanta,
    "GeoGraphyData": load_geography,
}
