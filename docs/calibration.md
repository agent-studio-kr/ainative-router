# Calibration set

2,179 queries built from the same public source benchmarks RouterArena uses, **with every RouterArena item removed**.
RouterArena (`full` 8,400 + `robustness` 420) is used only as an exclusion list and to reproduce each source's row
representation (so RouterArena's own prompt builder and scorers apply unchanged).

- Build: `python -m calib.build` (source loaders in `calib/sources/`), then `python -m calib.leakage --drop`.
- Manifest (IDs, sources, groups — no text): `data/calib/calib_manifest.jsonl`.
- Prompt formatting is byte-identical to RouterArena's `prep_datasets.py` on 7,000+ RouterArena rows
  (`tests/test_format_prompt.py`).
- Target size per config: RouterArena's per-config share of 2,000, minimum 20 (`calib/targets.json`).
- Items whose gold answer, placed in `\boxed{}`, does not score 1.0 with RouterArena's scorer are dropped
  (e.g. equation-form MATH answers).

## Leakage checks

| Check | Result |
|---|---|
| Normalized exact question match | 0 |
| Context-prefix match (non-template contexts) | 0 |
| all-MiniLM-L6-v2 cosine ≥ 0.85 on question + context (templates excluded) | 23 flagged → dropped |
| Source-ID level (in loaders) | NarrativeQA documents, FinQA documents, AsDiv passages, PubMedQA abstracts, SuperGLUE passages/sentence pairs, WMT sentence pairs, QANTA IDs, MMLU-Pro question IDs, AIME problem IDs, LiveCodeBench question IDs (+ release date), chess games (move-sequence prefix) |

Chess is exempt from the embedding check (move strings embed similarly across different games); it uses the
structured game-level check instead. Report: `data/calib/leakage_report.json`.

## Sources

| Config | Calibration source (repo : config : split) | License | Notes |
|---|---|---|---|
| MMLUPro | TIGER-Lab/MMLU-Pro : test | MIT | RouterArena items excluded by question_id |
| MMLU | cais/mmlu : formal_logic, management : test+validation+dev | MIT | |
| ArcMMLU | **substitute** cais/mmlu : 7 adjacent subjects : test | MIT | RouterArena uses its own English translation of the Chinese ArcMMLU; the Chinese original could contain untranslated RouterArena items, so it is not used |
| MedMCQA | openlifescienceai/medmcqa : validation | Apache-2.0 | |
| PubMedQA | qiaojin/PubMedQA : pqa_labeled | MIT | abstract-level exclusion |
| OpenTDB | opentdb.com API (cached: `data/calib/cache/opentdb_verified.json`) | CC BY-SA 4.0 | also excludes cosine ≥ 0.85 (6 RouterArena items were later edited upstream) |
| GeoBench | daven3/geobench : apstudy | Apache-2.0 | |
| MusicTheoryBench | m-a-p/MusicTheoryBench : test+dev | see card | |
| SocialiQA | allenai/social_i_qa : train | see card | context-level exclusion |
| Ethics_* | hendrycks/ethics : test_hard (+test for deontology) | MIT | |
| AIME | AI-MO/aimo-validation-aime, opencompass/AIME2025, di-zhang-fdu/AIME_1983_2024 (≤2021) | Apache-2.0 / MIT | problem-ID exclusion |
| MATH | EleutherAI/hendrycks_math : train | MIT | RouterArena uses MATH-500 |
| GSM8K | openai/gsm8k : main : train | MIT | RouterArena uses test |
| AsDiv | EleutherAI/asdiv : validation | CC BY-NC 4.0 | only split; passage-level exclusion |
| MathQA | allenai/math_qa : train | Apache-2.0 | RouterArena uses test |
| FinQA | dreamerdeo/finqa : all splits | see card (original repo MIT) | document-level exclusion |
| WMT19-*-en | wmt/wmt19 : validation | research use | sentence-pair exclusion |
| NarrativeQA | deepmind/narrativeqa : validation | Apache-2.0 | document-level split (RouterArena uses test) |
| SuperGLUE-* | aps/super_glue : copa / record / rte+axb / boolq / multirc / wic / wsc : train (axb: test) | per task | passage / sentence-pair exclusion |
| QANTA | community-datasets/qanta : guessdev+buzzdev | research use | qanta_id exclusion |
| GeoGraphyData | **substitute** mandarjoshi/trivia_qa : rc.nocontext : validation (geography filter) | research use | original "GeoGraphyData_100k" source not found |
| LiveCodeBench | lighteval/code_generation_lite : release_v6 increment (2025-01..04) | see card | RouterArena LCB items are all from release_v2; disjoint by date |
| ChessInstruct(_mcq) | Thytu/ChessInstruct : test | CC BY-4.0 | game-level exclusion |

## Extension for held-out validation (v2)

`python -m calib.build_extra --seed 11` draws 2,197 further items with the same loaders and targets, excluding
RouterArena **and** every source ID / question of the 2,179-query set; `calib.leakage --drop` then removed 20
(MiniLM ≥ 0.85), leaving 2,177 (`cal2_*` IDs). Merged set (4,356 rows): `data/calib_v2/calib_manifest.jsonl`,
leakage report `data/calib_v2/leakage_report.json`. The extension is used only to validate policies fit on the
original set, never to fit them.

A second extension (`--base data/calib_v2 --prefix cal3 --seed 23 --out data/calib_v3`; 2,081 drawn, 28 dropped,
2,053 kept) excludes both earlier sets and was used only for the pre-registered test in `docs/prereg_v3.md`.
Manifest: `data/calib_v3/calib_manifest.jsonl`.

## Known biases

- LiveCodeBench v6 increment is harder than RouterArena's release_v2 items (55% "hard"); absolute code accuracy is
  underestimated for every model, relative ranking is used.
- Minimum 20 items per config changes the mix; rows are re-weighted to RouterArena's per-config counts.
