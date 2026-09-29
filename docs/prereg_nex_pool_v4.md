# Pre-registration: adding nex-n2.5-pro to the option pool (2026-09-29)

Written before `nex-agi/nex-n2.5-pro` is run on the calibration set.

## Why this model

`nex-agi/nex-n2.5-pro` was not in the model list used for the earlier screens. On the same 140-item MMLU-Pro screen used for every other candidate (`data/probe_mmlupro_prompts.jsonl`, no RouterArena items), three settings were measured once: reasoning effort low 90.7% at $0.312 per 1K queries, default 87.9% at $0.284, and `nex-n2.5-mini` 83.5% at $0.165. For comparison, `deepseek-v4.1-flash` (low) scored 89.3% at $0.682 on the same items. The low setting is the one carried forward. This choice was made on the screen items only.

## Data and options

- Data: `data/calib_v4` (8,805 items, external sources only, same leakage exclusion as before). No RouterArena data, labels or proportions are used.
- A: the five options from `docs/prereg_mcq_split_v4.md` (`gemma-4-31b-it`, `gemini-3-flash-preview` reasoning off, `deepseek-v4.1-flash` low, `deepseek-v4-flash-0731` low, `deepseek-v4-flash-0731` default).
- B: A plus `nex-agi/nex-n2.5-pro` (reasoning effort low).
- Both arms use the same 11 content categories and the same items (items missing a result for any option are dropped from both).

## Decision

- Protocol: group 5-fold, classifier and bagged policy refit inside each fold, sources weighted equally, 1,000 group-bootstrap resamples, paired B − A (`실험/nex_pool_cv_v4.py`).
- **Adopt B only if the 95% CI lower bound of ΔArena > 0.** One evaluation, no other nex settings or pools tried on the calibration set.
- Every calibration row must have a real answer before the evaluation runs (failed calls are re-run, not dropped).
- If adopted: the submitted policy is B fitted on all of `data/calib_v4`; `nex-agi/nex-n2.5-pro` is added to `model_cost.json` at its OpenRouter list price ($0.075 input / $0.25 output per million tokens), and every RouterArena row routed to it is generated with a real call.
