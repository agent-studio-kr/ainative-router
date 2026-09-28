# Pre-registration: second held-out extension (v3 candidates)

Written and committed **before** the second extension (`data/calib_v3`, `cal3_*` IDs) is built or any model is run on it.

## Data

`python -m calib.build_extra --base data/calib_v2 --prefix cal3 --seed 23 --out data/calib_v3`, then
`calib.leakage --drop`. Excludes RouterArena, the 2,179-query fit set and the 2,177-query first extension (source IDs
and normalized questions). Models run on it: the six v2 pool models only.

## Candidates (fit on `data/calib_v2`, i.e. fit set + first extension, never on `cal3_*`)

| ID | Policy |
|---|---|
| A (baseline) | current v2 policy (`artifacts/policy.json`, frozen) |
| B | task-group cells, refit on all 4,354 v2 rows, six models |
| C | task-group × domain cells, refit on all 4,354 v2 rows, six models |

## Decision rule

On `cal3_*` rows only, with RouterArena per-config re-weighting and official prices: paired bootstrap (2,000
resamples, seed 0) of ΔArena = candidate − A. Two comparisons → Bonferroni: a candidate passes if the lower bound of
its **97.5%** two-sided interval (1.25th percentile) is > 0. If both pass, the one with the larger point ΔArena is
adopted. If neither passes, v2 stays. No other candidate is added after seeing `cal3_*` results.
