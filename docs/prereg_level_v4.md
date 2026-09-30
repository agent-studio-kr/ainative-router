# Pre-registration: predicted difficulty level inside each content category (2026-09-30)

Written before the confirmation split is scored. The confirmation split was not used to explore or select the structure below. Its items were part of earlier full-data CV tests of other hypotheses (subject-level multiple-choice split, `nex-n2.5-pro`, and a same-day exploratory embedder comparison); none of those involved difficulty levels. A plan and script review by a second model (Codex) was applied before running.

## Data split

`data/calib_v4` (external sources only, no RouterArena data) is split once by group, stratified by external source: 40% of each source's groups form the **confirmation split** (3,773 items), the rest is the **exploration split** (5,032 items). Seed 20260930, logic in `scripts/eval_prereg_level.py::confirmation_ids`.

## What was explored (exploration split only, group 5-fold CV)

- Predicting difficulty from the question body alone is weak but not zero: AUROC inside a category ≈ 0.55–0.75 for "the cheapest option answers wrong", ≈ 0.7–0.8 for "the reasoning option produces a long, costly trace".
- 26 sub-cell structures were compared against the 11-category policy: category × 2/3/4 predicted-difficulty bins for four difficulty targets, category × k-means clusters (K = 2/4/8), two feature sets, global vs per-category predictors, and one 2-D split. 24 of 26 had a positive CV ΔArena (+0.04 to +0.63); the largest came from ethics and math, none from the multiple-choice category. Because the best structure was selected from many, its exploration estimate (+0.63 [−0.00, +1.19]) is optimistic, which is why the decision below uses only the untouched confirmation split.

## Structure under test (B)

- Category: the existing content classifier (MiniLM body embedding → logistic regression, 11 categories).
- Level: inside each predicted category, a logistic regression predicts P(`gemma-4-31b-it` answers wrong) from the MiniLM embedding plus 7 surface features of the question body (length, word count, digit ratio, `$` rate, LaTeX commands, line count, operator rate). Three levels at the category's tertiles of the out-of-fold predictions.
- Policy: one option per (category, level) sub-cell, shrunk toward the category estimate (`policy.fit(use_domain=True)`), 25 group-bootstrap refits with a majority vote. Only sub-cells with ≥ 20 items in the original fitting sample are eligible (fixed before bagging); other sub-cells and categories without a difficulty model use the category choice. A single-class training fold of the difficulty model predicts that fold's positive rate.
- Options: the five options of `docs/prereg_mcq_split_v4.md` for both arms.

A: the same pipeline with no level (11 categories).

## Decision

- Fit A and B on the exploration split; score both once on the confirmation split. Sources weighted equally within each split (weights 1/n from that split's own counts, fixed across bootstrap resamples as in earlier pre-registrations); 1,000 group-bootstrap resamples of the confirmation split, paired B − A.
- **Adopt B only if the 95% CI lower bound of ΔArena > 0.** One run of `scripts/eval_prereg_level.py`; no other structures are scored on the confirmation split.
- If adopted: refit B on all of `data/calib_v4` (same settings) for the submission, and disclose the difficulty model in the method notes. If not adopted: the submitted policy is unchanged.
