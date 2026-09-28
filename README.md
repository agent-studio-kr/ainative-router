# Ko-Agent Router

> Submitted to RouterArena as `ko-agent-router` (v1 was submitted as `arena-router`; Python package name `arena_router`).

A task-aware LLM router for the [RouterArena](https://github.com/RouteWorks/RouterArena) benchmark.
Each query is mapped to a **task group** (from the structure of its instruction), and each task group is
mapped to the model with the best accuracy–cost trade-off, chosen on an **external calibration set that
contains no RouterArena items**.

## Results (RouterArena `full`, [PR #213](https://github.com/RouteWorks/RouterArena/pull/213))

| Version | Arena Score | Accuracy | Cost / 1K queries | Robustness | Opt.Sel | Opt.Cost | Opt.Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| **v2** (current, local official scripts) | **76.64** | 79.13% | $0.385 | 88.81 | 6.43 | 15.94 | 92.69 |
| v1 (official CI) | 76.25 | 78.64% | $0.379 | 90.24 | 7.51 | 16.33 | 92.93 |

v2 adds one model to the pool; it was accepted on a fresh held-out calibration extension **before** RouterArena
was routed (ΔArena +0.50, 95% CI [+0.27, +0.75] vs v1).

Details, cross-validation, and routing shares: [docs/results.md](docs/results.md).

## How it works

```
prompt ─► task group ─────────────► policy lookup ─► model
          · instruction head of the public RouterArena eval-config templates (26 groups)
          · if the head does not match (e.g. paraphrased instruction): kNN over calibration prompts
```

| Task group (examples) | Model |
|---|---|
| general multiple-choice (MMLU-Pro, OpenTDB, PubMedQA, …) | google/gemini-3-flash-preview |
| GSM8K / MATH, LiveCodeBench, NarrativeQA, one translation pair | google/gemma-4-31b-it |
| AIME / AsDiv, FinQA, cloze, two translation pairs | deepseek/deepseek-v4-flash-0731 |
| ethics | deepseek/deepseek-v4.1-flash |
| some reading comprehension | qwen/qwen3-235b-a22b-2507 |
| NLI, free-form chess | openai/gpt-6-luna |

The policy is chosen by sweeping λ in `argmax_model (accuracy − λ·cost)` per task group and keeping the
policy with the highest Arena Score on the calibration set. Full method: [docs/method.md](docs/method.md).

## Compliance with the RouterArena evaluation-only rule

- **No RouterArena query, answer, label, or other submission's predictions were used** to fit, tune, or
  design any router component. RouterArena data is used only as an *exclusion list* when building the
  calibration set, and for the final evaluation.
- Calibration set: 2,179 queries (policy fit) + a 2,177-query extension (held-out validation only), from the same
  public source benchmarks, with every RouterArena item removed
  (normalized exact match, template-stripped MiniLM cosine ≥ 0.85, and source-ID checks such as NarrativeQA
  documents, chess games, LiveCodeBench problem IDs, WMT sentence pairs). See [docs/calibration.md](docs/calibration.md).
- The task-group detector uses only the **public eval-config templates** shipped in RouterArena
  (`config/eval_config/zero-shot/*.json`), not RouterArena queries.
- Policy, signal code, kNN index, and external model revisions are **hash-frozen** in
  [`artifacts/FREEZE.json`](artifacts/FREEZE.json) before routing RouterArena; the router refuses to load on mismatch.
- Every prediction row records `requested_model`, `model_used`, `upstream_provider`, `request_id`,
  `invoked_at`, and the provider-reported token usage. No cascades or multi-call tricks: one call per query.

## Repository layout

```
arena_router/        router: signals, policy, freeze, RouterArena adapter glue, prediction filler
calib/               calibration set: source loaders, leakage check, OpenRouter client, scoring (RouterArena scorers)
tests/               unit + integration tests
artifacts/           policy.json, policy_report.json (CV), policy_holdout_report.json, FREEZE.json (+ *_v1 files)
data/calib/          calibration manifest (IDs only), per-model scores, signals, reports, OpenTDB cache
data/calib_v2/       fit set + held-out extension: manifest (IDs only), per-model scores, signals, reports
patches/             RouterArena changes (model registration, adapter, config)
scripts/             helpers (TLS shim, RouterArena runner, template paraphrases)
```

Calibration **question text is not redistributed** (mixed source licenses, some non-commercial); it is
regenerated deterministically from public sources by `calib.build` (see below).

## Reproduce

### Setup

```bash
git clone https://github.com/agent-studio-kr/arena-router && cd arena-router
uv sync
git clone https://github.com/RouteWorks/RouterArena third_party/RouterArena
git -C third_party/RouterArena checkout f371ef0
git -C third_party/RouterArena apply ../../patches/routerarena-ko-agent-router.patch
(cd third_party/RouterArena && ../../.venv/bin/python ../../scripts/ra_run.py scripts/process_datasets/prep_datasets.py)
```

Behind a TLS-intercepting proxy, add `--native-tls` to `uv` commands; `scripts/_tls.py` makes Python use the
system trust store.

### A. Verify routing and the freeze — no API cost

```bash
# 1. regenerate the calibration set from public sources (IDs must match data/calib/calib_manifest.jsonl)
uv run python -m calib.build
uv run python -m calib.leakage --drop
uv run python scripts/check_manifest.py
# held-out extension (v2 validation)
uv run python -m calib.build_extra --seed 11
CALIB_DATA_DIR=data/calib_v2 uv run python -m calib.leakage --drop
CALIB_DATA_DIR=data/calib_v2 uv run python scripts/check_manifest.py

# 2. rebuild the kNN index and check every frozen hash
uv run python -m arena_router.freeze --rebuild-index --verify        # -> "freeze OK"

# 3. route RouterArena and compare with the submitted predictions
cd third_party/RouterArena
../../.venv/bin/python ../../scripts/ra_run.py router_inference/generate_prediction_file.py ko-agent-router full
../../.venv/bin/python ../../scripts/ra_run.py router_inference/generate_prediction_file.py ko-agent-router robustness
../../.venv/bin/python ../../scripts/ra_run.py llm_evaluation/run.py ko-agent-router robustness   # -> 0.8881
```

Routing is deterministic: the `prediction` fields match the submitted files.

### B. Re-score the submitted generations — no API cost

Put the submitted `ko-agent-router.json` (with `generated_result`) in `router_inference/predictions/`, then:

```bash
../../.venv/bin/python ../../scripts/ra_run.py router_inference/check_config_prediction_files.py ko-agent-router full --check-generated-result
../../.venv/bin/python ../../scripts/ra_run.py tools/audit_token_accounting.py ko-agent-router --strict
../../.venv/bin/python ../../scripts/ra_run.py llm_evaluation/run.py ko-agent-router full    # -> 0.7664
```

### C. Rebuild everything from scratch — ~$20 of OpenRouter calls

```bash
echo "OPENROUTER_API_KEY=..." > .env
MODELS=google/gemma-4-31b-it,openai/gpt-6-luna,google/gemini-3-flash-preview,deepseek/deepseek-v4.1-flash,qwen/qwen3-235b-a22b-2507,deepseek/deepseek-v4-flash-0731
uv run python -m calib.infer --input data/calib/calib_prompts.jsonl --models $MODELS --concurrency 48   # ~$6
uv run python -m calib.score_results
uv run python -m arena_router.build_policy --cv 5 --no-domain --models $MODELS   # fit on the 2,179-query set
# held-out validation on the extension (~$8)
CALIB_DATA_DIR=data/calib_v2 uv run python -m calib.infer --input data/calib_v2/calib_prompts.jsonl --models $MODELS --concurrency 48
CALIB_DATA_DIR=data/calib_v2 uv run python -m calib.score_results
CALIB_DATA_DIR=data/calib_v2 uv run python scripts/holdout_eval.py artifacts/policy_v1.json   # -> artifacts/policy_holdout_report.json
uv run python -m arena_router.freeze                              # re-freeze (new FREEZE.json)
# then A.3, and fill generations with provenance (~$5):
uv run python -m arena_router.fill_predictions ko-agent-router
```

LLM outputs are sampled at provider defaults (as in RouterArena's OpenRouter client), so re-running C gives
close but not bit-identical accuracy.

## License

Code: Apache-2.0. Calibration data comes from third-party datasets under their own licenses
(listed in [docs/calibration.md](docs/calibration.md)); only IDs and derived scores are included here,
except the OpenTDB cache (CC BY-SA 4.0, © Open Trivia Database contributors).
