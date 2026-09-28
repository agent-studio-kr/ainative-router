# Method

## 1. Signals

**Task group.** RouterArena formats every query with a per-source zero-shot template from its public
eval configs (`config/eval_config/zero-shot/*.json`). Configs that share the same fixed instruction head form one
group — 26 groups for 41 configs (e.g. one general multiple-choice head is shared by 12 configs).

- Primary: longest matching instruction head (`arena_router/signals.py: template_heads`). Error ≤ 1% on the
  calibration set (`tests/signals/test_task_signal.py`).
- Fallback (head not found, e.g. paraphrased or typo'd instruction): majority vote of the 7 nearest calibration
  prompts, using all-MiniLM-L6-v2 on the first 300 characters (instruction-dominated). On calibration prompts whose
  instruction was paraphrased by an LLM (3 variants per head: paraphrase, synonym/grammar change, typos;
  `scripts/paraphrase_templates.py`), fallback accuracy is **92.9%** (n = 1,090, held-out half).

**Domain.** `llm-semantic-router/Vela-1.0-Encoder-307M-Domain` (14 MMLU-Pro domains) on the question body with the
template removed. Computed but **not used by the final policy** (see §3).

## 2. Policy

Cells: task group (optionally task group × domain). For each cell and model we estimate weighted mean accuracy and
cost on the calibration set, shrunk toward the parent cell: `est = (n·mean + K0·parent) / (n + K0)`, `K0 = 10`;
cells with `n < 20` inherit the parent's choice.

For λ on a log grid we pick `argmax_m (acc_m − λ·cost_m)` per cell and keep the policy with the highest Arena Score
(RouterArena's formula, β = 0.1) on the calibration set. Sweeping λ traces the accuracy–cost Pareto frontier of
per-cell policies; the Arena Score selects the operating point.

**Weights.** Calibration rows are re-weighted per source config to RouterArena's source-config proportions
(counts only), so aggregate accuracy/cost reflect the benchmark's mix.

**Costs** use RouterArena's price table (`model_cost/model_cost.json`) — the same prices the official scorer uses —
and OpenRouter list prices for models not yet in the table (registered at those prices in the patch).

## 3. Validation

Group 5-fold cross-validation (groups = same document / passage / game / problem family). In each fold the policy
and the best single model are chosen on the training folds and applied to the held-out fold; ΔArena CI by paired
bootstrap (2,000 resamples) over all held-out rows.

| Variant | CV Arena | Best single model (CV) | ΔArena [95% CI] |
|---|---:|---:|---|
| task group × domain | 77.28 | 76.59 | +0.69 [−0.22, +1.62] |
| **task group (selected)** | **77.48** | 76.59 | **+0.89 [+0.36, +1.42]** |

The domain split raised in-sample Arena (79.0) but lowered CV Arena, so it was dropped.

**v2 acceptance (held-out).** Model-pool changes are accepted only on a held-out calibration extension
(2,177 queries, `calib/build_extra.py`, same construction and leakage checks, disjoint from the fit set). The v2
policy (fit on the 2,179-query set) beat v1 there by ΔArena +0.50 [+0.27, +0.75] (`scripts/holdout_eval.py`).
Reports: `artifacts/policy_report.json`, `artifacts/policy_report_domain.json`.

## 4. Candidate models

v2 pool: the five v1 models + `deepseek/deepseek-v4-flash-0731` (registered in RouterArena's price table at its
OpenRouter list price, $0.021 / $0.32 per M tokens). The extra model came from a post-submission screen of 91
OpenRouter models (incl. reasoning-off / low-effort variants) on 140 non-RouterArena MMLU-Pro items, followed by
full-calibration-set runs of the shortlist.

v1: five OpenRouter models were measured on the full calibration set: google/gemma-4-31b-it, openai/gpt-6-luna,
google/gemini-3-flash-preview, deepseek/deepseek-v4.1-flash, qwen/qwen3-235b-a22b-2507. The shortlist came from a
cost/quality probe of 23 OpenRouter models on 166 non-RouterArena items (MMLU-Pro test items absent from RouterArena,
and AIME 2025 problems absent from RouterArena). Disclosure: the list of 23 probed candidates was informed by the
model pools of public leaderboard submissions, and during early exploration we looked at aggregate per-model
statistics in other submissions' public prediction files. No per-query RouterArena outcome was used, and the
selection among candidates and the policy itself rely only on the external probe and the calibration set.

## 5. Freeze and inference

`artifacts/FREEZE.json` stores SHA-256 of the policy, kNN index, and signal/policy/router code, plus Hugging Face
revisions of the two encoders. The router verifies it at load. Generation uses one OpenRouter call per query with
provider-default sampling (as RouterArena's own OpenRouter client), recording provenance per row.
