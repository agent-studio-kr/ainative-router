"""지식/MCQ 계열 보정 원천 로더.

RouterArena 가 쓴 원본 HF 원천에서 RouterArena 문항을 제외하고 같은 표현(dataset_name·options·answer·context·metadata)으로 CalibRow 를 만든다.
원천·표현 근거와 대체 원천은 data/calib/sources_knowledge.md 에 기록한다.

    LOADERS[config_name](target, seed, excl) -> list[CalibRow]   # 최대 ceil(1.3 × target) 개
"""
from __future__ import annotations

import base64
import difflib
import hashlib
import gzip
import json
import math
import random
import time
import urllib.request
from collections import defaultdict
from typing import Callable

from calib.common import DATA_DIR, ROOT, CalibRow, ExclusionIndex, normalize

OVERSAMPLE = 1.3
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TARGETS = json.loads((ROOT / "calib" / "targets.json").read_text())


# ---------------------------------------------------------------- 공통 헬퍼
def _n(target: int) -> int:
    return math.ceil(target * OVERSAMPLE)


def _rng(seed: int, name: str) -> random.Random:
    return random.Random(f"{seed}:{name}")


def _ra_subset_weights(config: str) -> dict[str, int]:
    """targets.json routerarena_subsets 에서 config 의 하위 Dataset name 별 RouterArena 문항 수."""
    return {k.split("|", 1)[1]: v for k, v in TARGETS["routerarena_subsets"].items() if k.split("|", 1)[0] == config}


def _allocate(n: int, weights: dict[str, int]) -> dict[str, int]:
    """최대 잉여 방식으로 n 을 weights 비율로 나눈다."""
    total = sum(weights.values())
    raw = {k: n * w / total for k, w in weights.items()}
    quota = {k: math.floor(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (quota[k] - raw[k], k))[: n - sum(quota.values())]:
        quota[k] += 1
    return quota


def _sample(rows: list[CalibRow], n: int, rng: random.Random) -> list[CalibRow]:
    rows = sorted(rows, key=lambda r: r.source_id)
    rng.shuffle(rows)
    return rows[:n]


def _sample_by_subset(rows: list[CalibRow], n: int, weights: dict[str, int], rng: random.Random) -> list[CalibRow]:
    """RouterArena 하위셋 비율로 n 개를 뽑는다. 가용 문항이 모자란 하위셋의 부족분은 남은 하위셋에 같은 비율로 재배분."""
    by_name: dict[str, list[CalibRow]] = defaultdict(list)
    for r in rows:
        by_name[r.dataset_name].append(r)
    cap = {k: len(by_name.get(k, [])) for k in weights}
    quota = {k: 0 for k in weights}
    while True:
        open_ = {k: w for k, w in weights.items() if quota[k] < cap[k]}
        left = n - sum(quota.values())
        if left <= 0 or not open_:
            break
        for k, q in _allocate(left, open_).items():
            quota[k] = min(cap[k], quota[k] + q)
    out: list[CalibRow] = []
    for name in sorted(quota):
        out += _sample(by_name.get(name, []), quota[name], rng)
    return out


def _keep(rows: list[CalibRow], excl: ExclusionIndex, config: str) -> list[CalibRow]:
    ids = excl.source_ids.get(config, set())
    return [r for r in rows if r.source_id not in ids and not excl.is_excluded(r.question, r.context)]


def _exclude_ids_by_text(excl: ExclusionIndex, config: str, rows: list[CalibRow], ra_names: list[str]) -> set[str]:
    """RouterArena 문항을 원천 행에 텍스트로 매핑해 원천 ID 를 excl.source_ids[config] 에 등록한다."""
    ra = {normalize(q) for name in ra_names for q in excl.questions_by_dataset.get(name, [])}
    ids = {r.source_id for r in rows if normalize(r.question) in ra}
    excl.source_ids.setdefault(config, set()).update(ids)
    return ids


def _load(repo: str, config: str | None, split: str, **kw):
    from datasets import load_dataset

    return load_dataset(repo, config, split=split, **kw)


def _hf_file(repo: str, path: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, path, repo_type="dataset")


def _ra_contexts(dataset_name: str) -> list[str]:
    """RouterArena 문항의 Context (ExclusionIndex 는 데이터셋별 context 를 노출하지 않아 HF 캐시에서 다시 읽는다)."""
    ra = _load("RouteWorks/RouterArena", None, "full")
    return [c for n, c in zip(ra["Dataset name"], ra["Context"]) if n == dataset_name and c]


def _qid(text: str) -> str:
    return hashlib.sha1(normalize(text).encode()).hexdigest()[:12]


# ---------------------------------------------------------------- MMLU-Pro
def load_mmlupro(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    """TIGER-Lab/MMLU-Pro test. dataset_name=MMLUPro_<category>, answer=answer_index 문자열."""
    weights = _ra_subset_weights("MMLUPro")
    rows = [
        CalibRow(
            dataset_name=f"MMLUPro_{r['category']}",
            config_name="MMLUPro",
            question=r["question"].strip(),  # RouterArena 는 MMLU-Pro 질문 앞뒤 공백을 제거해 저장
            answer=str(r["answer_index"]),
            options=list(r["options"]),
            source_hf="TIGER-Lab/MMLU-Pro:default:test",
            source_id=str(r["question_id"]),
            group_id=f"mmlupro:{r['src']}",
        )
        for r in _load("TIGER-Lab/MMLU-Pro", None, "test")
        if f"MMLUPro_{r['category']}" in weights
    ]
    _exclude_ids_by_text(excl, "MMLUPro", rows, list(weights))
    return _sample_by_subset(_keep(rows, excl, "MMLUPro"), _n(target), weights, _rng(seed, "MMLUPro"))


# ---------------------------------------------------------------- MMLU (+ ArcMMLU 대체)
def _mmlu_rows(subject: str, splits: list[str], dataset_name: str, config: str, letter: bool) -> list[CalibRow]:
    rows = []
    for split in splits:
        for i, r in enumerate(_load("cais/mmlu", subject, split)):
            rows.append(
                CalibRow(
                    dataset_name=dataset_name,
                    config_name=config,
                    question=r["question"],
                    answer=LETTERS[r["answer"]] if letter else str(r["answer"]),
                    options=list(r["choices"]),
                    source_hf=f"cais/mmlu:{subject}:{split}",
                    source_id=f"{subject}:{split}:{i}",
                    group_id=f"mmlu:{subject}",
                )
            )
    return rows


def load_mmlu(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    """cais/mmlu formal_logic·management. RouterArena 가 test 를 대부분 써서 test 잔여 + validation + dev 를 쓴다."""
    weights = _ra_subset_weights("MMLU")
    rows = []
    for name in weights:
        subject = name.removeprefix("MMLU_")
        rows += _mmlu_rows(subject, ["test", "validation", "dev"], name, "MMLU", letter=False)
    _exclude_ids_by_text(excl, "MMLU", rows, list(weights))
    return _sample_by_subset(_keep(rows, excl, "MMLU"), _n(target), weights, _rng(seed, "MMLU"))


# ArcMMLU 는 원본이 중국어(patrickshitou/ArcMMLU)이고 RouterArena 는 자체 영어 번역본을 쓴다.
# 번역 문항은 텍스트로 제외할 수 없고(누출 위험) LLM 번역도 금지이므로, 같은 형식(영어 4지선다·문자 정답)의
# 인접 분야 MMLU 과목으로 대체한다 (정보/데이터 과학·경영·정보보안 — ArcMMLU 의 문헌정보·데이터과학·정보·기록관리 범위에 가까운 과목).
ARCMMLU_SUBSTITUTE_SUBJECTS = [
    "college_computer_science",
    "high_school_computer_science",
    "computer_security",
    "machine_learning",
    "marketing",
    "public_relations",
    "business_ethics",
]


def load_arcmmlu(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    rows = []
    for subject in ARCMMLU_SUBSTITUTE_SUBJECTS:
        rows += _mmlu_rows(subject, ["test"], "ArcMMLU", "ArcMMLU", letter=True)
    return _sample(_keep(rows, excl, "ArcMMLU"), _n(target), _rng(seed, "ArcMMLU"))


# ---------------------------------------------------------------- 의학
def load_medmcqa(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    """openlifescienceai/medmcqa validation (RouterArena 와 같은 split). metadata={"subject_name"}."""
    rows = [
        CalibRow(
            dataset_name="MedMCQA",
            config_name="MedMCQA",
            question=r["question"],
            answer=str(r["cop"]),
            options=[r["opa"], r["opb"], r["opc"], r["opd"]],
            metadata={"subject_name": r["subject_name"]},
            source_hf="openlifescienceai/medmcqa:default:validation",
            source_id=r["id"],
            group_id=f"medmcqa:{r['id']}",
        )
        for r in _load("openlifescienceai/medmcqa", None, "validation")
    ]
    _exclude_ids_by_text(excl, "MedMCQA", rows, ["MedMCQA"])
    return _sample(_keep(rows, excl, "MedMCQA"), _n(target), _rng(seed, "MedMCQA"))


PUBMEDQA_OPTIONS = ["yes", "no", "maybe"]


def load_pubmedqa(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    """qiaojin/PubMedQA pqa_labeled. context=str(contexts 리스트) (RouterArena 표현), 같은 초록(pubid)은 같은 그룹."""
    rows = [
        CalibRow(
            dataset_name="PubMedQA",
            config_name="PubMedQA",
            question=r["question"],
            answer=str(PUBMEDQA_OPTIONS.index(r["final_decision"])),
            context=str(list(r["context"]["contexts"])),
            options=list(PUBMEDQA_OPTIONS),
            source_hf="qiaojin/PubMedQA:pqa_labeled:train",
            source_id=str(r["pubid"]),
            group_id=f"pubmed:{r['pubid']}",
        )
        for r in _load("qiaojin/PubMedQA", "pqa_labeled", "train")
    ]
    ids = _exclude_ids_by_text(excl, "PubMedQA", rows, ["PubMedQA"])
    # 같은 초록 문맥을 공유하는 문항도 제외
    ra_ctx = {normalize(r.context) for r in rows if r.source_id in ids}
    excl.source_ids["PubMedQA"].update(r.source_id for r in rows if normalize(r.context) in ra_ctx)
    return _sample(_keep(rows, excl, "PubMedQA"), _n(target), _rng(seed, "PubMedQA"))


# ---------------------------------------------------------------- 지리 · 음악 · 사회
def load_geobench(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    """daven3/geobench geobenchmark_apstudy.json (영어 5지선다, AP 지리·환경과학). answer=answerKey 인덱스."""
    data = json.loads(open(_hf_file("daven3/geobench", "geobenchmark_apstudy.json")).read())
    rows = []
    for i, r in enumerate(data):
        choices = r["question"]["choices"]
        labels = [c["label"] for c in choices]
        rows.append(
            CalibRow(
                dataset_name="GeoBench",
                config_name="GeoBench",
                question=r["question"]["stem"],
                answer=str(labels.index(r["answerKey"])),
                options=[c["text"] for c in choices],
                source_hf="daven3/geobench:apstudy:geobenchmark_apstudy.json",
                source_id=f"apstudy:{i:04d}",
                group_id=f"geobench:{_qid(r['question']['stem'])}",
            )
        )
    return _sample(_keep(rows, excl, "GeoBench"), _n(target), _rng(seed, "GeoBench"))


def load_musictheorybench(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    """m-a-p/MusicTheoryBench test(+dev). question=stem, options=A~D 순서, answer=인덱스."""
    rows = []
    for split in ["test", "dev"]:
        for r in _load("m-a-p/MusicTheoryBench", None, split):
            keys = sorted(k for k, v in r["options"].items() if v is not None)
            rows.append(
                CalibRow(
                    dataset_name="MusicTheoryBench",
                    config_name="MusicTheoryBench",
                    question=r["stem"],
                    answer=str(keys.index(r["answer"])),
                    options=[r["options"][k] for k in keys],
                    source_hf=f"m-a-p/MusicTheoryBench:default:{split}",
                    source_id=f"{split}:{r['id']}",
                    group_id=f"mtb:{r['subject']}:{split}:{r['id']}",
                )
            )
    _exclude_ids_by_text(excl, "MusicTheoryBench", rows, ["MusicTheoryBench"])
    return _sample(_keep(rows, excl, "MusicTheoryBench"), _n(target), _rng(seed, "MusicTheoryBench"))


def load_socialiqa(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    """allenai/social_i_qa (parquet 변환본) train. context=상황, answer=label-1."""
    rows = [
        CalibRow(
            dataset_name="SocialiQA",
            config_name="SocialiQA",
            question=r["question"],
            answer=str(int(r["label"]) - 1),
            context=r["context"],
            options=[r["answerA"], r["answerB"], r["answerC"]],
            source_hf="allenai/social_i_qa:default:train",
            source_id=f"train:{i}",
            group_id=f"siqa:{_qid(r['context'])}",
        )
        for i, r in enumerate(_load("allenai/social_i_qa", None, "train", revision="refs/convert/parquet"))
    ]
    # 질문 문장("What will Sasha want to do next?")은 흔해서 질문만으로 제외하면 과다 제외된다.
    # 대신 같은 상황(context)을 공유하는 문항은 RouterArena 문항과 한 그룹으로 보고 모두 제외한다.
    ra_ctx = {normalize(c) for c in _ra_contexts("SocialiQA")}
    excl.source_ids.setdefault("SocialiQA", set()).update(r.source_id for r in rows if normalize(r.context) in ra_ctx)
    kept = [r for r in rows if r.source_id not in excl.source_ids["SocialiQA"] and not excl.is_excluded(r.question, r.context)]
    return _sample(kept, _n(target), _rng(seed, "SocialiQA"))


# ---------------------------------------------------------------- ETHICS (hendrycks/ethics)
def _ethics_csv(sub: str, split: str):
    import pandas as pd

    return pd.read_csv(_hf_file("hendrycks/ethics", f"data/{sub}/{split}.csv"))


def _ethics_binary(sub: str, col: str, splits: list[str], target: int, seed: int, excl: ExclusionIndex):
    """commonsense/justice: options=['False','True'], answer=str(1-label) (RouterArena 표현: label 1=비윤리 → 'False')."""
    config = f"Ethics_{sub}"
    rows = []
    for split in splits:
        df = _ethics_csv(sub, split)
        for i, r in df.iterrows():
            rows.append(
                CalibRow(
                    dataset_name=config,
                    config_name=config,
                    question=str(r[col]),
                    answer=str(1 - int(r["label"])),
                    options=["False", "True"],
                    source_hf=f"hendrycks/ethics:{sub}:{split}",
                    source_id=f"{split}:{i}",
                    group_id=f"{config}:{_qid(str(r[col]))}",
                )
            )
    return _sample(_keep(rows, excl, config), _n(target), _rng(seed, config))


def load_ethics_commonsense(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    """test_hard (RouterArena 100문항 모두 test_hard, 긴 Reddit 시나리오 27% 포함)."""
    return _ethics_binary("commonsense", "input", ["test_hard"], target, seed, excl)


def load_ethics_justice(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    return _ethics_binary("justice", "scenario", ["test_hard"], target, seed, excl)


def _ethics_groups(sub: str, split: str) -> list[tuple[str, list[str], list[int], int]]:
    """고정 크기(deontology 4행, virtue 5행) 연속 행을 한 시나리오로 묶는다 → (scenario, options, labels, 첫 행 번호)."""
    size = 5 if sub == "virtue" else 4
    df = _ethics_csv(sub, split)
    groups: list[tuple[str, list[str], list[int], int]] = []
    for i, r in df.iterrows():
        if sub == "virtue":
            scenario, opt = str(r["scenario"]).split(" [SEP] ", 1)
            opt = opt.strip()  # RouterArena 는 virtue 특성어 공백을 제거
        else:
            scenario, opt = str(r["scenario"]), str(r["excuse"])
        if i % size == 0:
            groups.append((scenario, [], [], i))
        assert groups[-1][0] == scenario, f"ethics {sub}/{split} row {i}: group boundary mismatch"
        groups[-1][1].append(opt)
        groups[-1][2].append(int(r["label"]))
    return groups


def load_ethics_deontology(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    """시나리오당 변명 4개 중 합당(label=1)이 정확히 1개인 그룹만 4지선다로 (RouterArena 63문항이 모두 이 형태).
    test_hard 에 이런 그룹이 70개뿐이라(63개 RouterArena 사용) test split 의 같은 형태 그룹을 함께 쓴다."""
    config = "Ethics_deontology"
    rows = []
    for split in ["test_hard", "test"]:
        for scenario, opts, labels, first in _ethics_groups("deontology", split):
            if sum(labels) != 1:
                continue
            rows.append(
                CalibRow(
                    dataset_name=config,
                    config_name=config,
                    question=scenario,
                    answer=str(labels.index(1)),
                    options=opts,
                    source_hf=f"hendrycks/ethics:deontology:{split}",
                    source_id=f"{split}:{first}",
                    group_id=f"{config}:{_qid(scenario)}",
                )
            )
    return _sample(_keep(rows, excl, config), _n(target), _rng(seed, config))


def load_ethics_virtue(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    """시나리오당 특성 5개(정답 1개) → 5지선다. RouterArena 는 보기 순서를 섞으므로 seed 로 섞는다."""
    config = "Ethics_virtue"
    rng = _rng(seed, config + ":options")
    rows = []
    for scenario, opts, labels, first in _ethics_groups("virtue", "test_hard"):
        order = list(range(len(opts)))
        rng.shuffle(order)
        rows.append(
            CalibRow(
                dataset_name=config,
                config_name=config,
                question=scenario,
                answer=str(order.index(labels.index(1))),
                options=[opts[j] for j in order],
                source_hf="hendrycks/ethics:virtue:test_hard",
                source_id=f"test_hard:{first}",
                group_id=f"{config}:{_qid(scenario)}",
            )
        )
    return _sample(_keep(rows, excl, config), _n(target), _rng(seed, config))


# ---------------------------------------------------------------- OpenTDB (opentdb.com API, 캐시)
OPENTDB_CACHE = DATA_DIR / "cache" / "opentdb_verified.json.gz"  # gzip: 1.3MB → 수백 KB


def _get_json(url: str, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.loads(resp.read())
        except (OSError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(10 * (attempt + 1))
    raise AssertionError("unreachable")


def _b64(s: str) -> str:
    return base64.b64decode(s).decode("utf-8")


def fetch_opentdb(force: bool = False) -> list[dict]:
    """opentdb.com 의 검증된 전체 문항을 받아 캐시한다 (API 제한: 5초당 1요청 → 약 10분).
    HF 미러가 없어 RouterArena 와 같은 원천(opentdb.com API)을 직접 쓴다. 캐시가 재현성의 기준이다."""
    if OPENTDB_CACHE.exists() and not force:
        return json.loads(gzip.decompress(OPENTDB_CACHE.read_bytes()))
    base = "https://opentdb.com"
    cats = _get_json(f"{base}/api_category.php")["trivia_categories"]
    out: list[dict] = []
    for cat in cats:
        token = _get_json(f"{base}/api_token.php?command=request")["token"]
        remaining = _get_json(f"{base}/api_count.php?category={cat['id']}")["category_question_count"][
            "total_question_count"
        ]
        while remaining > 0:
            time.sleep(5.5)
            res = _get_json(
                f"{base}/api.php?amount={min(50, remaining)}&category={cat['id']}&encode=base64&token={token}"
            )
            code = res["response_code"]
            if code == 5:  # rate limit
                continue
            if code in (1, 4):  # 남은 문항 없음
                break
            if code != 0:
                raise RuntimeError(f"opentdb response_code={code} for category {cat}")
            for q in res["results"]:
                out.append(
                    {
                        "category": _b64(q["category"]),
                        "type": _b64(q["type"]),
                        "difficulty": _b64(q["difficulty"]),
                        "question": _b64(q["question"]),
                        "correct_answer": _b64(q["correct_answer"]),
                        "incorrect_answers": [_b64(a) for a in q["incorrect_answers"]],
                    }
                )
            remaining -= len(res["results"])
    out.sort(key=lambda q: (q["category"], q["question"]))
    OPENTDB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    OPENTDB_CACHE.write_bytes(gzip.compress(json.dumps(out, ensure_ascii=False, indent=0).encode(), mtime=0))
    return out


def load_opentdb(target: int, seed: int, excl: ExclusionIndex) -> list[CalibRow]:
    """dataset_name=OpenTDB_<category>, 보기=정답+오답 섞기(RouterArena 도 섞음), metadata={difficulty,type,original_category}."""
    weights = _ra_subset_weights("OpenTDB")
    rng = _rng(seed, "OpenTDB:options")
    rows = []
    for q in fetch_opentdb():
        name = f"OpenTDB_{q['category']}"
        if name not in weights:
            continue
        opts = [q["correct_answer"], *q["incorrect_answers"]]
        rng.shuffle(opts)
        rows.append(
            CalibRow(
                dataset_name=name,
                config_name="OpenTDB",
                question=q["question"],
                answer=str(opts.index(q["correct_answer"])),
                options=opts,
                metadata={"difficulty": q["difficulty"], "type": q["type"], "original_category": q["category"]},
                source_hf="opentdb.com:api:verified",
                source_id=_qid(q["question"]),
                group_id=f"opentdb:{_qid(q['question'])}",
            )
        )
    # OpenTDB 는 RouterArena 수집 이후 오타 수정 등으로 문항이 편집됐다 (RouterArena 887 중 6개가 현재 원문과 불일치).
    # 편집된 RouterArena 문항이 새 텍스트로 섞여 들지 않도록 유사도 0.85 이상이면 같은 문항으로 보고 제외한다.
    ra = [normalize(q) for name in weights for q in excl.questions_by_dataset.get(name, [])]
    excl.source_ids.setdefault("OpenTDB", set()).update(r.source_id for r in rows if _near_any(normalize(r.question), ra))
    return _sample_by_subset(_keep(rows, excl, "OpenTDB"), _n(target), weights, _rng(seed, "OpenTDB"))


def _near_any(text: str, candidates: list[str], threshold: float = 0.85) -> bool:
    for c in candidates:
        m = difflib.SequenceMatcher(None, text, c)
        if m.real_quick_ratio() >= threshold and m.quick_ratio() >= threshold and m.ratio() >= threshold:
            return True
    return False


LOADERS: dict[str, Callable[[int, int, ExclusionIndex], list[CalibRow]]] = {
    "MMLUPro": load_mmlupro,
    "MMLU": load_mmlu,
    "ArcMMLU": load_arcmmlu,
    "MedMCQA": load_medmcqa,
    "PubMedQA": load_pubmedqa,
    "OpenTDB": load_opentdb,
    "GeoBench": load_geobench,
    "MusicTheoryBench": load_musictheorybench,
    "SocialiQA": load_socialiqa,
    "Ethics_commonsense": load_ethics_commonsense,
    "Ethics_deontology": load_ethics_deontology,
    "Ethics_justice": load_ethics_justice,
    "Ethics_virtue": load_ethics_virtue,
}
