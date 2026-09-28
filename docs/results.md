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
