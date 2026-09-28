# Results

## v2 (current submission)

v2 = v1's five models + `deepseek/deepseek-v4-flash-0731`, fit on the same 2,179-query calibration set. Five task
groups (≈3.4% of RouterArena's weight) change model: AIME/AsDiv (gemma → 0731), FinQA (qwen → 0731),
SuperGLUE cloze (v4.1-flash → 0731), WMT19 kk-en (gemini → 0731), lt-en (luna → 0731). Multiple-choice stays on
gemini-3-flash.

**Acceptance (pre-registered bar: 95% CI lower bound of ΔArena > 0 vs v1), decided before routing RouterArena.**
A new 2,177-query calibration extension (same construction, disjoint from RouterArena and from the fit set) was
built and every candidate model was run on it. Neither v1 nor v2 saw these rows:

| Held-out extension (2,175 rows scored) | Arena | Accuracy | $/1K |
|---|---:|---:|---:|
| v1 | 77.20 | 80.0% | 0.433 |
| **v2** | **77.71** | 80.6% | 0.438 |
| Δ (paired bootstrap, 2,000 resamples) | **+0.50 [+0.27, +0.75]** | | |

Report: `artifacts/policy_holdout_report.json` (`scripts/holdout_eval.py`).

**RouterArena official CI** (PR #213, `ko-agent-router`): Arena **76.57**, accuracy 79.04%, $0.385 / 1K, robustness
88.81, Opt.Sel / Opt.Cost / Opt.Acc 6.43 / 15.80 / 92.40, 0 abnormal entries. The first CI attempt was aborted by a
runner shutdown mid-scoring; the re-run completed. Local vs CI differ by 0.09 pp accuracy (LiveCodeBench timing).

**RouterArena, local official scripts** (RouterArena `f371ef0` + patch):

| | v1 | v2 |
|---|---:|---:|
| Arena Score | 76.32 | **76.64** |
| Accuracy | 78.72% | **79.13%** |
| Cost / 1K | $0.379 | $0.385 |
| Robustness | **90.24** | 88.81 |
| Opt.Sel / Opt.Cost / Opt.Acc | 7.51 / 16.47 / 93.23 | 6.43 / 15.94 / 92.69 |
| Abnormal rows (8,400 regular) | 0 | 0 |

v2 routing share (8,400 regular rows): gemini-3-flash 79.9%, gemma-4-31b 10.8%, deepseek-v4-flash-0731 3.3%,
deepseek-v4.1-flash 2.6%, qwen3-235b 1.8%, gpt-6-luna 1.6%.

Robustness drops because math word problems are now split across two models (AIME/AsDiv → 0731, GSM8K/MATH →
gemma): a paraphrased prompt that the kNN fallback assigns to the neighbouring group now changes model.
Two of the 4,045 optimality-only rows (0731 on LiveCodeBench_105 and MMLUPro_math_7577) never completed within
5 × 15 min and are recorded as failed generations; they do not enter the Arena Score.

Other options considered on the same data and **not** adopted:

| Option | Evidence | Why not |
|---|---|---|
| Refit on all 4,354 rows (fit + extension) | CV +0.37 vs v1-refit | The multiple-choice cell (66% weight) flips between gemini / luna / v4.1-flash across folds and seeds (in-sample Arena within 0.08); no held-out set left to confirm the refit |
| + gemma-4-31b `reasoning: high` arm | held-out +0.46 (vs +0.50 without) | No gain; not expressible in the prediction format |
| task × domain cells | CV +0.38 [−0.16, +0.91] vs task-only | CI includes 0 |

## v1 — official CI (RouterArena PR [#213](https://github.com/RouteWorks/RouterArena/pull/213))

| Arena | Accuracy | Cost / 1K | Robustness | Opt.Sel | Opt.Cost | Opt.Acc |
|---:|---:|---:|---:|---:|---:|---:|
| **76.25** | 78.64% | $0.379 | 90.24 | 7.51 | 16.33 | 92.93 |

The local run below differs by 0.08 pp accuracy (~7 queries), most likely code-execution timing on LiveCodeBench
items (Linux CI vs. macOS); cost and robustness are identical.

## v1 — local run

RouterArena official scripts run locally on RouterArena commit `f371ef0` + the v1 patch (submission ID `arena-router`).

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

## v1 routing share (8,400 regular rows)

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

v2: `artifacts/FREEZE.json` (policy, code, kNN index). v1's record is kept as `artifacts/FREEZE.v1.json`, its policy
and CV report as `artifacts/policy_v1.json` / `artifacts/policy_report_v1.json`.

v1:

`artifacts/FREEZE.prev.json` is the first freeze. After routing RouterArena, a docstring in
`arena_router/policy.py` was edited (comment-only; no behavior change) and the artifacts were re-frozen
(`artifacts/FREEZE.json`). Policy, kNN index, and all other hashes are identical between the two records.

## Post-submission exploration before v2

All of the following were evaluated **on the calibration set only** (no RouterArena data), after the v1 submission,
on the original 2,179 rows. None cleared the pre-registered bar (paired-bootstrap 95% CI lower bound of ΔArena > 0
vs. v1). The last row motivated building the held-out extension that accepted v2 (above).

| Idea | Calibration result | Decision |
|---|---|---|
| Agreement cascade (gemma + qwen agree → keep, else gemini) | +0.38 Arena, −41% cost | Not adopted: RouterArena bills each row at one model's price, so probe calls cannot be reported honestly |
| Prompt-only "needs the stronger model" predictor (MiniLM + domain, logistic) | CV AUROC 0.501 | Not adopted: no signal |
| Screening 91 OpenRouter models on 140 non-RouterArena MMLU-Pro items, incl. reasoning-off / low-effort variants of GLM-5.3, Kimi-K2.x, Qwen3.8, MiniMax, DeepSeek | Best single MCQ models on the full calibration MCQ group: deepseek-v4.1-flash 89.8%, gemini-3-flash 89.7% | No cheaper model matched gemini-3-flash on multiple-choice |
| v2 policy with 4 more arms (gpt-6-luna-pro, gemma-4-31b@high reasoning, deepseek-v4-flash-0731, mimo-v2.6-pro) | ΔArena +0.25 [−0.06, +0.56] | Not adopted: CI includes 0 |
