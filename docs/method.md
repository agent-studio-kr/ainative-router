# Method

Content-only routing: **question body → content category → model**. The router never reads RouterArena files and
does not match any instruction or prompt-template text.

> Earlier versions (v1/v2, tag [`template-v2`](https://github.com/agent-studio-kr/ainative-router/tree/template-v2))
> grouped queries by RouterArena's eval-config instruction heads. After review on RouterArena PR #213 this was
> identified as dataset-specific routing (ruling in RouterArena #140) and was removed together with the re-weighting
> to RouterArena's per-config counts.

## 1. Signal

**Question body (structural rule).** The prompt is split on blank lines; the first paragraph (instruction) and the
last paragraph (answer format) are dropped when there are at least three paragraphs, otherwise the whole prompt is
kept (`arena_router/signals.py: question_body`). Format cues are then removed generically — short line-leading
labels (`Xxx:`), option letters (`A)`, `(b)`), and placeholder lines (`None`) — so only the question, passage and
option *content* remains (`routing_text`). No string is compared against any benchmark template.

**Content category.** all-MiniLM-L6-v2 embedding of the body → multinomial logistic regression (C = 10), trained
only on the external calibration items (§3). Labels are our own 11-category taxonomy of the external sources:
multiple-choice knowledge, multiple-choice math, math word/competition problems, finance math, code, translation,
reading comprehension, NLI / word sense, ethics, chess, trivia (`CATEGORY_OF_SOURCE`). The source name is used
only as a training label, never at routing time.

Group 5-fold CV accuracy of the category classifier: **85.0%**. With the format cues left in it was 94.0%; the
difference is how much a classifier would lean on formatting, which is why the cues are removed.

Tests (`tests/signals/test_content_signal.py`): routing is invariant to replacing the first and last paragraphs
with arbitrary text and to relabelling fields / option letters, and no router module references benchmark files.

## 2. Policy

Cells = predicted categories. For each cell and model: weighted mean accuracy and cost on the calibration set, shrunk
toward the global mean (`est = (n·mean + K0·global) / (n + K0)`, `K0 = 10`). For λ on a log grid,
`argmax_m (acc_m − λ·cost_m)` per cell; the λ with the highest Arena Score (RouterArena formula, β = 0.1) on the
calibration set is kept.

Policy rows use **out-of-fold** predicted categories, so the policy is fit on the categories the classifier would
actually assign.

**Weights: uniform per external source** (each source's rows sum to 1). No RouterArena proportions are used.

**Costs** use RouterArena's price table (`model_cost/model_cost.json`) and OpenRouter list prices for models not in it
(registered at those prices in the patch).

## 3. Calibration data

6,409 external queries (fit set 2,179 + two extensions 2,177 and 2,053) from the same public source benchmarks with
every RouterArena item removed (see [calibration.md](calibration.md)). Six OpenRouter models were run on all of them.

## 4. Validation

Group 5-fold CV (groups = same document / passage / game / problem family). In each outer fold the classifier, the
out-of-fold categories, the policy and the best single model are all refit on the training part and applied to the
held-out part. ΔArena CI by paired bootstrap (2,000 resamples). Report: `artifacts/policy_report.json`.

| | CV Arena | Accuracy | $/1K |
|---|---:|---:|---:|
| **Router** | **72.15** | 74.3% | 0.487 |
| Best single model per fold | 69.24 | 72.8% | 1.361 |
| Δ [95% CI] | **+2.91 [+2.19, +3.61]** | | |

(Absolute values are lower than RouterArena's because every source counts equally, including the hardest ones.)

## 5. Freeze and inference

`artifacts/FREEZE.json` stores SHA-256 of the policy, the classifier weights and the signal/policy/router code, plus
the Hugging Face revision of the embedding model. The router verifies it at load. Generation uses one OpenRouter call
per query with provider-default sampling, recording provenance per row.
