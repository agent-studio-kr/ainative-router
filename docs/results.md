# Results

## Current submission: content-only router with reasoning settings (`ainative-router`)

Adds reasoning-setting options (see method §2). On calibration CV the Arena score is unchanged (72.43 vs 72.43) at
22% lower cost ($0.377 vs $0.481 per 1K).

**Official CI (PR #213): Arena 75.42, accuracy 78.28%, $0.535 / 1K, robustness 83.10, 0 abnormal entries.**

Further options tested on calibration data and not adopted (no significant CV gain): reasoning caps
(`reasoning.max_tokens` 1,000), gpt-6-luna reasoning levels and luna-only sub-splits of multiple-choice, cheap-default
gate, length-based difficulty split, 1-SE rule, stronger embeddings, soft (probability-weighted) routing, kNN routing.

### Previous content-only version (6 models, default reasoning)

RouterArena official scripts, run locally (RouterArena `f371ef0` + `patches/routerarena-ainative-router.patch`).
The official CI result is posted on [PR #213](https://github.com/RouteWorks/RouterArena/pull/213).

| Metric | Value |
|---|---:|
| Arena Score | **75.48** |
| Accuracy | 78.39% |
| Cost per 1K queries (RouterArena price table) | $0.547 |
| Robustness (1 − flip rate) | 85.00 |
| Opt.Sel / Opt.Cost / Opt.Acc | 4.24 / 22.24 / 89.79 |
| Failed generations among 8,400 regular rows | 0 (after re-requesting 6 empty / errored responses) |

- `check_config_prediction_files.py --check-generated-result`: all checks passed.
- `tools/audit_token_accounting.py --strict`: 0 unaccounted reasoning tokens.
- The first CI run (Arena 75.41) had 6 rows with empty responses (provider-side `finish_reason: error`, one
  whitespace-only answer, one runaway generation). Empty responses are now re-requested like errors; the numbers
  above are from before that fix and are superseded by the CI result on the PR.

Routing share (8,400 regular rows): deepseek-v4.1-flash 71.1%, gemini-3-flash 14.1%, deepseek-v4-flash-0731 9.8%,
gemma-4-31b 5.0%.

Calibration CV (all 6,409 external rows, sources weighted equally): router 72.15 vs best single model 69.24,
ΔArena +2.91 [95% CI +2.19, +3.61]; category classifier accuracy 85.0% (`artifacts/policy_report.json`).

## History

| Version | Grouping | RouterArena Arena | Status |
|---|---|---:|---|
| v1 `arena-router` | RouterArena instruction templates | 76.25 (CI) | withdrawn |
| v2 `ko-agent-router` | same + one more model | 76.57 (CI) | withdrawn |
| **current `ainative-router`** | **question content only** | 75.48 (local) | submitted |

v1/v2 keyed on the fixed instruction heads of RouterArena's eval configs and re-weighted calibration rows to
RouterArena's per-config counts. The RouterArena maintainers identified this as dataset-specific routing (see
RouterArena #140); both were removed. The old code, reports and held-out analyses are kept under tag
[`template-v2`](https://github.com/agent-studio-kr/ainative-router/tree/template-v2).
