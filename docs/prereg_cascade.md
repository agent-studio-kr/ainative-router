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

## Result (2026-09-29, recorded after the run)

Rejected. Group 5-fold CV vs the current policy (sources weighted equally), ΔArena [95% CI]:
C=1000 −0.15 [−0.65, +0.37], 1500 −0.27 [−0.72, +0.23], 2000 −0.18 [−0.55, +0.22], 3000 −0.09 [−0.47, +0.25], 4000 −0.17 [−0.51, +0.14].

Within the multiple-choice category alone the cascade halves the cost (C=1000: $0.281 → $0.153 per 1K) for −0.39 points of accuracy, but truncated questions answered with reasoning off drop from 59.8% to 46.2% correct, and queries routed to the reasoning model that genuinely need long reasoning (math-like items) lose more. An exploratory follow-up (cascade on the v4.1 option only, policy fixed — defined after seeing the breakdown) was also flat: C=3000 +0.01 [−0.16, +0.17], C=1000 −0.12 [−0.43, +0.16]. The current policy stays.
