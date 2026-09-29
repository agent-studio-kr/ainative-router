# Pre-registration: same-model length-capped cascade (2026-09-29)

Written before measuring the fallback on the calibration set.

## Rule under test

For every query routed to `deepseek-v4.1-flash` (reasoning effort low) or `deepseek-v4-flash-0731` (reasoning effort low):

1. Call the model with `max_tokens = C` (a hard generation limit, unlike the reasoning budget, which providers do not enforce).
2. If the response has no usable answer (empty content or `finish_reason = length`), call **the same model** again with reasoning disabled and submit that answer.
3. The submitted row bills **both calls**: `token_usage` is the sum of the two calls' input and output tokens, priced at the same model's `model_cost.json` price (both calls use the same model, so the sum is the exact cost). Both request ids are recorded.

No RouterArena data is used. The decision to cascade is made at inference time from the model's own output length only.

## Decision

- Candidates: C ∈ {1000, 1500, 2000, 3000, 4000}.
- Evaluation on the external calibration set only (`data/calib_v3`), same protocol as the policy: group 5-fold, bagged policy refit inside each fold, sources weighted equally, 1,000 group-bootstrap resamples, paired against the current policy.
- The capped first call is simulated exactly from the recorded full-length calibration trajectory (a generation limit truncates the same trajectory at C tokens; cost = input + C output tokens, no answer). The fallback answer is measured with real calls (`@off`) on every calibration query whose recorded reasoning exceeds 1,000 tokens.
- C is chosen as the candidate with the best CV Arena. **Adopt only if the 95% CI lower bound of ΔArena vs the current policy is > 0.** Otherwise the current policy stays.
- If adopted, RouterArena rows are generated with real cascade calls (no simulation).
