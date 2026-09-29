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
