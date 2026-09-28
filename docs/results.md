# Results

## Official CI (RouterArena PR [#213](https://github.com/RouteWorks/RouterArena/pull/213))

| Arena | Accuracy | Cost / 1K | Robustness | Opt.Sel | Opt.Cost | Opt.Acc |
|---:|---:|---:|---:|---:|---:|---:|
| **76.25** | 78.64% | $0.379 | 90.24 | 7.51 | 16.33 | 92.93 |

The local run below differs by 0.08 pp accuracy (~7 queries), most likely code-execution timing on LiveCodeBench
items (Linux CI vs. macOS); cost and robustness are identical.

## Local run

RouterArena official scripts run locally on RouterArena commit `f371ef0` + `patches/routerarena-arena-router.patch`.

## RouterArena `full` (8,400 queries) and `robustness` (420)

| Metric | Value |
|---|---:|
| Arena Score | **76.32** |
| Accuracy | 78.72% |
| Cost per 1K queries (RouterArena price table) | $0.379 |
| Robustness (1 − flip rate) | **90.24** |
| Opt.Sel / Opt.Cost / Opt.Acc (679 of 809 sub_10 queries with an optimal model) | 7.51 / 16.47 / 93.23 |
| Abnormal rows (failed generation, scored 0) | 0 |

- `check_config_prediction_files.py --check-generated-result`: all checks passed, no retired/redirected slugs.
- `tools/audit_token_accounting.py --strict`: 0 failed-inference rows, 0 unaccounted reasoning tokens.

## Calibration CV vs RouterArena

| | Arena | Accuracy | $/1K |
|---|---:|---:|---:|
| Calibration, group 5-fold CV | 77.48 | 80.2% | 0.41 |
| RouterArena `full` | 76.32 | 78.7% | 0.38 |

The policy was frozen before RouterArena was routed and was not changed after seeing RouterArena results.

## Routing share (8,400 regular rows)

| Model | Share |
|---|---:|
| google/gemini-3-flash-preview | 79.9% |
| google/gemma-4-31b-it | 12.3% |
| deepseek/deepseek-v4.1-flash | 3.3% |
| qwen/qwen3-235b-a22b-2507 | 2.7% |
| openai/gpt-6-luna | 1.8% |

## Single models on the calibration set (in-sample, re-weighted)

| Model | Accuracy | $/1K | Arena |
|---|---:|---:|---:|
| google/gemini-3-flash-preview | 79.7% | 0.544 | 76.59 |
| deepseek/deepseek-v4.1-flash | 79.7% | 1.183 | 75.16 |
| google/gemma-4-31b-it | 74.2% | 0.099 | 73.86 |
| openai/gpt-6-luna | 74.1% | 0.253 | 72.80 |
| qwen/qwen3-235b-a22b-2507 | 70.5% | 0.044 | 71.13 |

## Freeze history

`artifacts/FREEZE.prev.json` is the first freeze. After routing RouterArena, a docstring in
`arena_router/policy.py` was edited (comment-only; no behavior change) and the artifacts were re-frozen
(`artifacts/FREEZE.json`). Policy, kNN index, and all other hashes are identical between the two records.

## Post-submission exploration (not adopted)

All of the following were evaluated **on the calibration set only** (no RouterArena data), after the v1 submission.
None cleared the pre-registered bar (paired-bootstrap 95% CI lower bound of ΔArena > 0 vs. v1), so v1 stands.

| Idea | Calibration result | Decision |
|---|---|---|
| Agreement cascade (gemma + qwen agree → keep, else gemini) | +0.38 Arena, −41% cost | Not adopted: RouterArena bills each row at one model's price, so probe calls cannot be reported honestly |
| Prompt-only "needs the stronger model" predictor (MiniLM + domain, logistic) | CV AUROC 0.501 | Not adopted: no signal |
| Screening 91 OpenRouter models on 140 non-RouterArena MMLU-Pro items, incl. reasoning-off / low-effort variants of GLM-5.3, Kimi-K2.x, Qwen3.8, MiniMax, DeepSeek | Best single MCQ models on the full calibration MCQ group: deepseek-v4.1-flash 89.8%, gemini-3-flash 89.7% | No cheaper model matched gemini-3-flash on multiple-choice |
| v2 policy with 4 more arms (gpt-6-luna-pro, gemma-4-31b@high reasoning, deepseek-v4-flash-0731, mimo-v2.6-pro) | ΔArena +0.25 [−0.06, +0.56] | Not adopted: CI includes 0 |
