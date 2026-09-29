# Pre-registration: larger multiple-choice calibration and subject-level split (2026-09-29)

Written before any model is run on the new items.

## Data

`data/calib_v4` = `data/calib_v3` + new multiple-choice items (`cal4_` prefix) built with `calib.build_extra --base data/calib_v3 --prefix cal4 --seed 21 --scale 2.0 --configs MMLUPro,MedMCQA,PubMedQA,OpenTDB,GeoBench,ArcMMLU,SocialiQA,SuperGLUE-CausalReasoning`. Existing calibration items and every RouterArena item are excluded by the same leakage checks as before. No RouterArena data, labels or proportions are used beyond exclusion.

## Options measured on the new items

The options the current policy can pick for multiple-choice-like queries: `gemini-3-flash-preview` (reasoning off; the provider default is also non-thinking, so this run stands in for both gemini options), `gemma-4-31b-it`, `deepseek-v4.1-flash` (low), `deepseek-v4-flash-0731` (low), `deepseek-v4-flash-0731` (default). The comparison below uses only these options for both arms.

## Hypothesis and decision

- A (current): 11 content categories.
- B: the multiple-choice category split into the four MMLU super-categories (STEM / humanities / social sciences / other), mapping fixed in `실험/mcq_split_cv.py` before this run.
- Protocol: group 5-fold, classifier and bagged policy refit inside each fold, sources weighted equally, 1,000 group-bootstrap resamples, paired B − A.
- **Adopt B only if the 95% CI lower bound of ΔArena > 0.** Otherwise keep A. One evaluation, no re-running with other splits or weights.

## Result (2026-09-29, recorded after the run)

Rejected. All 2,396 new items were answered by all five options (11,980 calls, 0 invalid rows; calls that failed on transient credit errors were re-run until every row had a real answer). Group 5-fold CV on 8,768 items (37 older rows lack a result for one option):

| | accuracy | $/1K | Arena |
|---|---|---|---|
| A (11 categories), sources weighted equally | 74.08% | 0.322 | 72.53 |
| B (14 categories), sources weighted equally | 74.12% | 0.333 | 72.52 |

ΔArena (B − A) −0.02 [−0.37, +0.32] with sources weighted equally; −0.24 [−0.45, −0.03] with items weighted equally (reference only). The CI lower bound is below 0, so A stays and the submitted policy is unchanged.

On the new items alone, accuracy / $ per 1K: gemma 84.8% / 0.066, gemini (off) 90.0% / 0.452, v4.1 (low) 90.0% / 0.417, 0731 (low) 89.4% / 0.295, 0731 (default) 89.1% / 0.347.
