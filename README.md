# AINative Router

> Submitted to RouterArena as `ainative-router` ([PR #213](https://github.com/RouteWorks/RouterArena/pull/213)).
> Python package name: `arena_router`.

A content-only LLM router for the [RouterArena](https://github.com/RouteWorks/RouterArena) benchmark. It looks at
the **question body only**, predicts a content category with a classifier trained on external data, and sends each
category to the model with the best accuracy–cost trade-off on an **external calibration set that contains no
RouterArena items**. The router never reads RouterArena files and never matches instruction or template text.

## Results (RouterArena `full`)

| Arena Score | Accuracy | Cost / 1K queries | Robustness | Opt.Sel | Opt.Cost | Opt.Acc |
|---:|---:|---:|---:|---:|---:|---:|
| **75.48** (local official scripts) | 78.39% | $0.547 | 85.00 | 4.24 | 22.24 | 89.79 |

Details and history: [docs/results.md](docs/results.md).

## How it works

```
prompt ─► question body ─► content category ─► policy lookup ─► model
          drop first / last paragraph      MiniLM embedding + logistic regression,
          strip line labels, option        trained on external calibration questions
          letters, "None" placeholders     (11 categories, CV accuracy 85%) 
```

| Content category | Model |
|---|---|
| multiple-choice knowledge, ethics, NLI / word sense (default) | deepseek/deepseek-v4.1-flash |
| math word / competition problems, finance math, reading comprehension | deepseek/deepseek-v4-flash-0731 |
| code, multiple-choice math | google/gemma-4-31b-it |
| trivia, translation, chess | google/gemini-3-flash-preview |

Per category, λ is swept in `argmax_model (accuracy − λ·cost)` and the policy with the highest Arena Score on the
calibration set is kept; every external source has equal weight. Group 5-fold CV: router vs best single model
ΔArena **+2.91 [95% CI +2.19, +3.61]**. Full method: [docs/method.md](docs/method.md).

## Compliance with the RouterArena evaluation-only rule

- **Routing uses the query content only.** No RouterArena template strings, config files, or dataset identifiers are
  used by the router; `tests/signals/test_content_signal.py` checks that routing is invariant to rewriting the
  instruction / answer-format paragraphs and field labels, and that no router module references benchmark files.
- **No RouterArena query, answer, label, proportion, or other submission's predictions** is used to fit any
  component. RouterArena data is used only as an exclusion list when building the calibration set, and for the
  final evaluation. Policy weights are uniform per external source.
- Calibration set: 6,409 queries from the same public source benchmarks with every RouterArena item removed
  (normalized exact match, MiniLM cosine ≥ 0.85 on question + context, and source-ID checks). See
  [docs/calibration.md](docs/calibration.md).
- Policy, classifier weights, and code are **hash-frozen** in [`artifacts/FREEZE.json`](artifacts/FREEZE.json)
  before routing RouterArena; the router refuses to load on mismatch.
- Every prediction row records `requested_model`, `model_used`, `upstream_provider`, `request_id`,
  `invoked_at`, and the provider-reported token usage. One call per query, no cascades.
- History: v1/v2 grouped queries by RouterArena's instruction templates. That was dataset-specific routing
  (RouterArena #140) and was removed after review; the old code is kept under tag
  [`template-v2`](https://github.com/agent-studio-kr/ainative-router/tree/template-v2).

## Repository layout

```
arena_router/        router: content signal, policy, freeze, prediction filler
calib/               calibration set: source loaders, leakage check, OpenRouter client, scoring (RouterArena scorers)
tests/               unit + integration tests
artifacts/           policy.json, category_clf.npz, policy_report.json (CV), FREEZE.json
data/calib*/         calibration manifests (IDs only), per-model scores, reports, OpenTDB cache
patches/             RouterArena changes (model registration, adapter, config)
scripts/             helpers (TLS shim, RouterArena runner, manifest check)
```

Calibration **question text is not redistributed** (mixed source licenses, some non-commercial); it is
regenerated deterministically from public sources by `calib.build` (see below).

## Reproduce

### Setup

```bash
git clone https://github.com/agent-studio-kr/ainative-router && cd ainative-router
uv sync
git clone https://github.com/RouteWorks/RouterArena third_party/RouterArena
git -C third_party/RouterArena checkout f371ef0
git -C third_party/RouterArena apply ../../patches/routerarena-ainative-router.patch
(cd third_party/RouterArena && ../../.venv/bin/python ../../scripts/ra_run.py scripts/process_datasets/prep_datasets.py)
```

Behind a TLS-intercepting proxy, add `--native-tls` to `uv` commands; `scripts/_tls.py` makes Python use the
system trust store.

### A. Verify the freeze and routing — no API cost

```bash
uv run python -m arena_router.freeze --verify        # -> "freeze OK"
cd third_party/RouterArena
../../.venv/bin/python ../../scripts/ra_run.py router_inference/generate_prediction_file.py ainative-router full
../../.venv/bin/python ../../scripts/ra_run.py router_inference/generate_prediction_file.py ainative-router robustness
../../.venv/bin/python ../../scripts/ra_run.py llm_evaluation/run.py ainative-router robustness   # -> 0.8500
```

### B. Re-score the submitted generations — no API cost

Put the submitted `ainative-router.json` (with `generated_result`) in `router_inference/predictions/`, then:

```bash
../../.venv/bin/python ../../scripts/ra_run.py router_inference/check_config_prediction_files.py ainative-router full --check-generated-result
../../.venv/bin/python ../../scripts/ra_run.py tools/audit_token_accounting.py ainative-router --strict
../../.venv/bin/python ../../scripts/ra_run.py llm_evaluation/run.py ainative-router full    # -> 0.7548
```

### C. Rebuild everything from scratch — ~$30 of OpenRouter calls

```bash
# calibration sets from public sources (IDs must match the manifests)
uv run python -m calib.build && uv run python -m calib.leakage --drop && uv run python scripts/check_manifest.py
uv run python -m calib.build_extra --seed 11
CALIB_DATA_DIR=data/calib_v2 uv run python -m calib.leakage --drop
uv run python -m calib.build_extra --base data/calib_v2 --prefix cal3 --seed 23 --out data/calib_v3
CALIB_DATA_DIR=data/calib_v3 uv run python -m calib.leakage --drop
CALIB_DATA_DIR=data/calib_v3 uv run python scripts/check_manifest.py

echo "OPENROUTER_API_KEY=..." > .env
MODELS=google/gemma-4-31b-it,openai/gpt-6-luna,google/gemini-3-flash-preview,deepseek/deepseek-v4.1-flash,qwen/qwen3-235b-a22b-2507,deepseek/deepseek-v4-flash-0731
CALIB_DATA_DIR=data/calib_v3 uv run python -m calib.infer --input data/calib_v3/calib_prompts.jsonl --models $MODELS --concurrency 64
CALIB_DATA_DIR=data/calib_v3 uv run python -m calib.score_results
CALIB_DATA_DIR=data/calib_v3 uv run python -m arena_router.build_policy   # classifier + policy + CV report
uv run python -m arena_router.freeze
# then A, and fill generations with provenance:
uv run python -m arena_router.fill_predictions ainative-router
```

LLM outputs are sampled at provider defaults (as in RouterArena's OpenRouter client), so re-running C gives close
but not bit-identical accuracy.

## License

Code: Apache-2.0. Calibration data comes from third-party datasets under their own licenses
(listed in [docs/calibration.md](docs/calibration.md)); only IDs and derived scores are included here,
except the OpenTDB cache (CC BY-SA 4.0, © Open Trivia Database contributors).
