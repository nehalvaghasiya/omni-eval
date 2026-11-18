# OmniEval

[![Core CI](https://github.com/nehalvaghasiya/omni-eval/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/nehalvaghasiya/omni-eval/actions/workflows/ci.yml)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](#installation)
[![MIT license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/nehalvaghasiya/omni-eval/blob/main/LICENSE)

Compare prompts, models, and retrieval changes on the test cases that matter to your application. OmniEval scores saved outputs, shows what improved or failed, and saves reports you can review or check in CI.

Bring your questions and reference answers, export predictions from your existing workflow, and evaluate both versions locally. The same CLI and Python API also support retrieval results and observed agent traces.

**Development preview · 0.2.0.** See [project status and limitations](#project-status-and-limitations) for the current support boundaries.

## Contents

- [Why OmniEval](#why-omnieval)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Evaluate your own data](#evaluate-your-own-data)
- [Python API](#python-api)
- [Configuration](#configuration)
- [CLI reference](#cli-reference)
- [Metrics reference](#metrics-reference)
- [Results and reports](#results-and-reports)
- [Compare runs and enforce CI gates](#compare-runs-and-enforce-ci-gates)
- [Advanced usage](#advanced-usage)
- [Development and verification](#development-and-verification)
- [Troubleshooting](#troubleshooting)
- [Project status and limitations](#project-status-and-limitations)
- [Contributing and support](#contributing-and-support)
- [License](#license)

## Why OmniEval

- **Compare a prompt or model change.** Score two sets of predictions against the same references, compare paired results, and inspect failed cases.
- **Check retrieval and agent behavior.** Measure ranked results, supplied RAG evidence, observed state, and tool calls with task-specific metrics.
- **Use outputs you already have.** Read JSONL, mapped CSV, or optional Parquet files without a model API key or changes to your inference code.
- **See scores and coverage together.** Missing predictions, failed scoring, and ineligible cases remain visible alongside each score.
- **Make the evaluation repeatable.** Save reports, configuration, provenance, and metric versions; enforce score and coverage thresholds in CI.

## Installation

OmniEval supports **Python 3.11, 3.12, and 3.13**. The repository defaults to Python 3.12 and uses **uv** for dependencies, environments, commands, and builds. CI pins uv 0.12.1.

[Install uv](https://docs.astral.sh/uv/getting-started/installation/), then clone and install from source:

```bash
git clone https://github.com/nehalvaghasiya/omni-eval.git
cd omni-eval
uv sync --locked --no-dev
```

The committed `uv.lock` supplies resolved versions. `uv sync` creates an editable installation in `.venv`; no manual environment activation is required.

Choose the profile you need:

| Profile | Command | Includes |
|---|---|---|
| Runtime | `uv sync --locked --no-dev` | Core package and native metrics |
| Local API | `uv sync --locked --no-dev --extra api` | FastAPI and Uvicorn |
| Parquet | `uv sync --locked --no-dev --extra parquet` | PyArrow input support |
| Statistics references | `uv sync --locked --no-dev --extra stats` | NumPy/scikit-learn reference checks; native metrics work without them |
| Core development | `uv sync --locked --group dev` | Core package and developer tools |
| Full development | `uv sync --locked --group dev --all-extras` | Developer tools and all three extras |

Repeat `--extra` to combine extras, or use `--all-extras`. A subsequent `uv sync` reconciles the environment to the selected profile. The commands below use `uv run --no-sync` after setup to use that prepared environment.

Installation may need network access. Native evaluation of saved outputs runs without model downloads or provider requests.

## Quick start

Compare a prompt change for a support bot. Evaluate both versions against the same four questions, inspect the remaining failure, and save a report for each run.

The [support QA example](examples/support-qa) keeps reference questions in `cases.jsonl` and each version's answers in `baseline.jsonl` and `candidate.jsonl`. Run this from the repository root after installation:

```bash
uv run --no-sync python - <<'PY'
from omni_eval import compare, evaluate

baseline = evaluate("examples/support-qa/baseline.yaml", out="runs")
candidate = evaluate("examples/support-qa/candidate.yaml", out="runs")
comparison = compare(baseline, candidate)

print(f"Baseline answer match: {baseline.metric('answer_match').value:.0%}")
print(f"Candidate answer match: {candidate.metric('answer_match').value:.0%}")
print(f"Change: {100 * comparison.deltas['answer_match']:+.0f} percentage points")
print("Cases to review:", [point.id for point in candidate.failures(metric="answer_match")])
print("Baseline run:", baseline.path)
print("Candidate run:", candidate.path)
PY
```

Key output, followed by the saved run paths:

```text
Baseline answer match: 50%
Candidate answer match: 75%
Change: +25 percentage points
Cases to review: ['cancel']
```

The candidate fixes the refund and shipping answers but breaks cancellation. Open `report.html` in either printed run directory to review the scores and coverage. Use those directories with the CLI [comparison and CI gates](#compare-runs-and-enforce-ci-gates).

These are illustrative saved answers. Replace them with outputs from your own prompts or models to use the same workflow. `answer_match` compares short answers after normalizing case, punctuation, and whitespace; it does not judge semantic correctness or factual faithfulness.

## Evaluate your own data

Keep your evaluation questions and references in `cases.jsonl`, with one JSON object per line:

```jsonl
{"id":"refund","input":"How long do I have to request a refund? Answer with the time window only.","expected":"30 days"}
{"id":"cancel","input":"Where can I cancel my subscription? Answer with the location only.","expected":"account settings"}
{"id":"shipping","input":"How long does standard shipping take? Answer with the delivery time only.","expected":"2 business days"}
{"id":"password","input":"What should I use to reset my password? Answer with the method only.","expected":"reset link"}
```

Export the current version's answers to `baseline.jsonl`:

```jsonl
{"id":"refund","actual":"14 days"}
{"id":"cancel","actual":"account settings"}
{"id":"shipping","actual":"5 business days"}
{"id":"password","actual":"reset link"}
```

Export the changed prompt or model's answers to `candidate.jsonl`:

```jsonl
{"id":"refund","actual":"30 days"}
{"id":"cancel","actual":"contact support"}
{"id":"shipping","actual":"2 business days"}
{"id":"password","actual":"reset link"}
```

These records match the quick start files. Substitute your application's questions, references, and predictions while keeping the same IDs across versions.

For CLI evaluation, save `eval.yaml` beside the data:

```yaml
version: 1
task: text
data: cases.jsonl
predictions: candidate.jsonl
metrics: [answer_match]
```

```bash
uv run --no-sync omni-eval run -c eval.yaml --out runs
```

This saves the candidate's scores, coverage, and HTML report. Change `predictions` to `baseline.jsonl` to evaluate the current version. The [Python API](#python-api) below evaluates and compares both files directly. Dataset, prediction, rubric, and output paths in YAML resolve relative to the configuration file.

Each case needs a unique, nonempty string `id`. Optional fields include `input`, `expected`, `actual`, `probs`, `contexts`, `trace`, `meta`, `timing`, `error`, and `extensions`. Required evidence depends on the selected metric.

Predictions join by ID, never by row position. Duplicate and unknown IDs are rejected. Missing predictions remain visible through coverage. You can also include `actual` directly in each case for a single-file evaluation. To explicitly replace inline predictions with a separate file, set `actual_source: predictions`.

Choose the task and metrics for your workflow. All 11 task presets have example datasets and configurations in [examples](examples):

| Task | Input and purpose |
|---|---|
| `text` | Reference answers, generated text, JSON, and format constraints |
| `retrieval` | Ranked context IDs with explicit relevance judgments |
| `rag` | Answers, supplied contexts, citations, and retrieval evidence |
| `agent`, `mcp` | Observed state and tool-call traces |
| `classification` | Expected/predicted labels and optional class probabilities |
| `audio` | Reference and predicted transcripts |
| `vision` | Supplied labels; no image inference |
| `security` | Explicit scenario or policy oracle results |
| `code` | Python syntax validation |
| `sql` | Conservative single-SELECT filtering |

## Python API

The Python API uses the same scoring engine as the CLI. After creating the three JSONL files above, compare versions without writing YAML:

```python
from omni_eval import compare, evaluate, load, metrics

baseline = evaluate(
    "cases.jsonl", predictions="baseline.jsonl", task="text", metrics=["answer_match"], out="runs"
)
candidate = evaluate(
    "cases.jsonl", predictions="candidate.jsonl", task="text", metrics=["answer_match"], out="runs"
)
print(compare(baseline, candidate, seed=42).deltas)  # {'answer_match': 0.25}
print([point.id for point in candidate.failures(metric="answer_match")])  # ['cancel']

restored = load(candidate.path)
print(restored.summary)
print(metrics(task="text"))
```

Use `uv run --no-sync python your_script.py` to run a saved script. `evaluate` also accepts a `Config` from `omni_eval.types.config` or an iterable of `Case` objects. `Case` and `Run` are exported from `omni_eval`. Empty datasets are rejected.

## Configuration

Version 1 configurations reject unknown top-level fields. Start with `task` and `data`, then select metrics and optional behavior:

| Fields | Purpose |
|---|---|
| `version`, `name`, `task` | Configuration version, evaluation name, and task preset |
| `data`, `predictions`, `actual_source` | Dataset and optional separate predictions |
| `metrics` | Metric names; omitted/empty selects task defaults |
| `columns`, `json_columns` | CSV mapping from canonical field names to column names, and fields containing JSON |
| `output` | Saved-run directory and case-record retention |
| `limits` | Metric-specific settings under `metric_settings` |
| `slices` | Grouped comparisons using case metadata |
| `source`, `target` | `saved` (default), `http`, `openai`, or `replay` and explicit endpoint settings |
| `judge` | Rubric, model, consent, endpoint, and finite budgets for a configured judge |
| `plugin`, `suite` | Optional provider/suite selection, subject to capability availability |

CSV needs an explicit `columns` mapping, such as `{id: case_id, expected: reference, actual: prediction}`. Use `json_columns` for canonical fields containing structured JSON. Parquet needs the `parquet` extra and uses canonical case fields.

Set `output: {dir: runs, retain: false}` to save summaries without case, point, or error records. Replay then requires another retained source. For `source: replay`, set `data` to a saved run directory.

## CLI reference

Prefix commands with `uv run --no-sync` after installation:

| Command | Purpose |
|---|---|
| `omni-eval --help` / `--version` | Inspect commands or installed version |
| `omni-eval init PATH --task TASK` | Create editable configuration and starter cases |
| `omni-eval run -c eval.yaml --out runs` | Evaluate and save results |
| `omni-eval metrics list` / `metrics show NAME` | Discover metric names and requirements |
| `omni-eval report RUN --format html` | Render a saved run |
| `omni-eval compare BASELINE CANDIDATE` | Compare compatible saved runs |
| `omni-eval check RUN -c gates.yaml --strict` | Enforce metric and coverage thresholds |
| `omni-eval doctor` | Inspect the package and registry without checking endpoints |
| `omni-eval plugins list` / `show NAME` / `doctor NAME` / `setup NAME` | Inspect optional provider status |

Use command-specific `--help` for options. `init my-eval --task text` creates `cases.jsonl` and `eval.yaml` without overwriting existing files; replace the starter cases with your own data. For debugging, `omni-eval demo TASK` scores a synthetic task preset, and `run --dry-run` substitutes synthetic inputs and fixture workers without contacting the configured provider or scoring the real dataset.

## Metrics reference

`uv run --no-sync omni-eval metrics list` lists 64 registered names, including aliases and k templates. Use a concrete name such as `recall@5` or `topk@2` in configurations. `uv run --no-sync omni-eval metrics show recall@5` shows tasks, prerequisites, version, unit, and direction.

| Family | Registered names | Meaning and prerequisites |
|---|---|---|
| Classification | `accuracy`, `balanced_accuracy`, `confusion`, `mcc`, `kappa` | Expected/predicted labels. Accuracy is the fraction correct; balanced accuracy averages recall over reference classes. Confusion is a matrix, with no scalar value. MCC and kappa measure agreement. Constant single-class kappa is undefined. |
| Corpus F1 | `macro_f1`, `micro_f1`, `weighted_f1` | Whole-corpus confusion counts. Macro averages classes, micro pools counts, weighted uses reference support. |
| Corpus precision | `precision`, `macro_precision`, `micro_precision`, `weighted_precision` | TP/(TP+FP), using the selected averaging policy. Plain precision needs `positive_label`. |
| Corpus recall | `recall`, `macro_recall`, `micro_recall`, `weighted_recall` | TP/(TP+FN). Plain recall needs `positive_label`; zero denominators use zero. |
| Probabilities | `brier`, `logloss`, `topk@k`, `auroc`, `auprc` | Labeled probabilities/scores plus expected labels; a separate predicted label is unnecessary. Brier sums squared error over the full class vector (binary range 0–2). Log loss clips at 1e-15 and uses natural logs. Top-k breaks ties by declared class order. AUROC uses half credit for ties; AUPRC is threshold-block average precision. Both need `positive_label`; AUROC needs both reference classes, AUPRC needs a positive. |
| Calibration | `ece`, `mce` | Softmax confidence versus correctness, in equal-width bins (default 10; configurable 2–100). ECE weights bin gaps by support; MCE takes the largest gap. Lower is better. |
| Text overlap | `exact`, `answer_match`, `token_f1`, `token_precision`, `token_recall` | Exact is case-sensitive equality. Answer match uses Unicode NFKC, lowercasing, word tokens, and whitespace normalization. Token scores use multiset overlap. Expected and actual text are required. |
| JSON and format | `json_valid`, `schema_valid`, `fields_f1`, `format` | Strict JSON, local JSON Schema validation, exact JSON-typed key/value F1, and configured length/regex checks. Field matching ignores object key order. Schema validation requires a schema; remote references are unavailable. |
| Token likelihood | `nll`, `perplexity` | Actual output must contain nonempty finite, nonpositive token `logprobs`. NLL is negative mean log probability over all tokens; perplexity is exp(NLL), a dimensionless ratio. Overflow is unavailable rather than capped. |
| Transcripts | `wer`, `cer` | Mean per-case word/character edit distance divided by reference length. Empty/empty scores zero; nonempty prediction with an empty reference is undefined. Values can exceed one. |
| Retrieval | `precision@k`, `recall@k`, `hit@k`, `success@k`, `mrr`, `map`, `ndcg@k`, `rprec` | Ranked unique context IDs and explicit relevance judgments. P@k divides by k even for short lists; recall divides by all relevant IDs. MRR uses the first relevant rank, MAP uses relevant-rank precision, nDCG uses binary gains with log2 discounts. Unparameterized MRR/MAP inspect the full list. k is 1–1000; rank templates default to 5, top-k to 1. |
| RAG | `citation_valid`, `evidence_coverage`, `context_precision`, `rerank_delta` | Citation validity checks IDs in supplied contexts. Evidence coverage is relevant IDs retained / reference relevant IDs; context precision is relevant / selected contexts. Rerank delta compares nDCG@5 with ranked IDs in `extensions.before`. These are not factual-faithfulness judgments. |
| Agents and MCP | `state_success`, `state_partial`, `tool_allowed`, `tool_success`, `loop_rate`, `step_order`, `retry_rate` | Observed state must satisfy expected state. Tool checks require observed calls, permissions, success flags, or required order. Loop rate counts identical consecutive requests; retry rate is retries per tool call. Final self-reported success is insufficient. |
| Security | `attack_success`, `policy_violation` | Explicit verified scenario/policy oracle results. Detector flags and generated claims are insufficient. Lower is better. |
| Code and SQL | `code_valid`, `sql_select_safe` | Python AST syntax and a conservative single-SELECT text filter. Neither executes the prediction or establishes functional correctness. |
| Timing and reliability | `latency_mean`, `latency_p50`, `latency_p95`, `latency_p99`, `success_rate` | Latency is milliseconds with linear quantiles and matching scope/provenance. Success rate counts all attempted cases, including errors. Uploaded timing is not a fresh measurement. |

Unless stated otherwise, match/compliance scores are fractions in [0,1] and higher is better. MCC/kappa can be negative; loss, edit error, latency, violation, loop, and retry metrics are minimized. `rerank_delta` is signed. Empty or absent context lists are currently unscorable where context prerequisites apply; they are reported with reduced coverage.

Settings belong under `limits.metric_settings`, keyed by the exact requested name. For example, a probability evaluation at the repository root can use:

```yaml
version: 1
task: classification
data: examples/classification/cases.jsonl
metrics: [accuracy, auroc, ece]
limits:
  metric_settings:
    auroc:
      positive_label: B
    ece:
      bins: 10
```

`schema_valid` accepts `schema` in its metric settings or `expected.schema`. `format` accepts `min_length`, `max_length`, and a short `pattern`. Only local schema references such as `#/$defs/name` are resolved. Format regex checks are intended for trusted configurations, not an adversarial-input sandbox.

## Results and reports

Every metric result includes `value`, `status`, `n_total`, `n_eligible`, `n_scored`, `n_failed`, `n_missing`, and coverage (`n_scored / n_total`). High performance on a small eligible subset does not establish full-dataset performance. Undefined or failed scores use null and carry reason evidence. Corpus metrics need not have per-case scalar scores.

A saved run contains:

```text
runs/
└── <run-id>/
    ├── run.json          # Run identity, status, and configuration
    ├── manifest.json     # Provenance and metric versions
    ├── scores.json       # Summary scores and denominators
    ├── cases.jsonl       # Retained input/prediction records
    ├── points.jsonl      # Retained per-case metric evidence
    ├── errors.jsonl      # Retained errors
    ├── report.html       # Rendered HTML summary
    └── checksums.json    # Artifact integrity checks
```

The three JSONL files are omitted when `output.retain` is false. Loading checks artifact integrity and rejects altered, missing, added, incomplete, or symlinked files. Checksums detect file changes; they are not signatures or proof of data authenticity.

Reports render existing scores without rerunning metrics. Replace `runs/your-run-id` with the `Saved:` path printed by `run`:

```bash
RUN_DIR="runs/your-run-id"
uv run --no-sync omni-eval report "$RUN_DIR" --format html > report.html
uv run --no-sync omni-eval report "$RUN_DIR" --format json > report.json
uv run --no-sync omni-eval report "$RUN_DIR" --format md > report.md
uv run --no-sync omni-eval report "$RUN_DIR" --format junit > report.xml
```

Synthetic results are identified in human reports, JSON, and JUnit.

## Compare runs and enforce CI gates

Use the two saved run paths printed by the quick start, or compatible runs from your own evaluation:

```bash
BASELINE="runs/baseline-run-id"
CANDIDATE="runs/candidate-run-id"
uv run --no-sync omni-eval compare "$BASELINE" "$CANDIDATE" --metric answer_match
```

Comparison pairs stable IDs and checks inputs, references, metric settings, versions, units, sources, and latency methodology. Bootstrap intervals are deterministic for a fixed seed and require paired sample scores. Corpus-only metrics do not receive fabricated intervals. Slices use metadata keys and suppress groups below the minimum support (default 5).

Require every short answer in your regression set to match its reference. Save `gates.yaml` (also included in [the support QA example](examples/support-qa/gates.yaml)):

```yaml
gates:
  - metric: answer_match
    op: gte
    value: 1.0
    min_coverage: 1.0
```

```bash
uv run --no-sync omni-eval check "$CANDIDATE" -c gates.yaml --strict
```

The quick start candidate fails this gate because its cancellation answer is wrong, even though its overall score improved. Set thresholds for your evaluation's requirements. Gates report pass, fail, or inconclusive. Use `--strict` in CI so missing evidence cannot silently pass.

| Exit | Meaning |
|---|---|
| `0` | Normal completion; also inconclusive gates when `--strict` is omitted |
| `1` | Failed threshold |
| `2` | Invalid input or configuration |
| `3` | Runtime/provider failure |
| `4` | Inconclusive gate with `--strict` |
| `5` | Policy refusal |

`min_pairs` on an absolute saved-run gate is inconclusive because that command has no pairing evidence. Native scoring policy is version 2; older saved runs remain readable, but comparisons reject differing metric versions.

## Advanced usage

### HTTP and OpenAI-compatible sources

`source: http` and `source: openai` acquire predictions from an explicitly configured target. Configure `url_env`, optional `token_env`, allowed hosts/ports, and a timeout under `target`. OpenAI-compatible targets also require a model.

HTTP request mappings use safe `$.field` paths into the case. Response mappings use the same syntax; the default is `$.actual`. Requests honor configured retries and record measured timing. Acquisition is currently serial.

Use environment variables for credentials. [.env.example](https://github.com/nehalvaghasiya/omni-eval/blob/main/.env.example) lists the endpoint/token names; OmniEval does not automatically load an `.env` file. HTTPS and allowlist checks apply, redirects are refused, and local HTTP requires explicit opt-in.

### Optional rubric judge

`judge:relevance` provides an estimated rubric score through an explicitly configured OpenAI-compatible endpoint. A configuration saved at the repository root can use this template after replacing the host and model with your own values:

```yaml
version: 1
task: text
data: cases.jsonl
metrics: ["judge:relevance"]
judge:
  model: your-judge-model
  rubric: examples/judge/rubric.yaml
  url_env: OMNIEVAL_JUDGE_URL
  token_env: OMNIEVAL_JUDGE_TOKEN
  allowed_hosts: [judge.example.com]
  max_calls: 10
  max_tokens: 5120
  per_call_tokens: 512
  approve_external_data: true
```

Set `OMNIEVAL_JUDGE_URL` to your endpoint's full chat-completions URL. Approval permits forwarding required case fields to that endpoint. Keep credentials out of the configuration. The example [rubric](https://github.com/nehalvaghasiya/omni-eval/blob/main/examples/judge/rubric.yaml) defines the criterion, anchors, and required fields.

Local fake services exercise consent, response validation, evidence, errors, and budgets. Backend model quality remains unverified; a judge score is not ground truth.

### Plugin availability

`plugins list/show/doctor/setup` expose current capability status. Providers marked `planned` refuse real execution. `fixture:echo` demonstrates the worker protocol and always marks its result synthetic. There are no qualified upstream framework, scanner, GPU, or benchmark integrations.

### Local read-only API

Install the `api` extra, then save `local_api.py`:

```python
from omni_eval.server.app import create_app

app = create_app("runs")
```

```bash
uv run --no-sync uvicorn local_api:app --host 127.0.0.1 --port 8000
```

The API exposes `GET /health`, `/metrics`, `/runs`, and `/runs/{run_id}`. It has no production authentication and should be bound to loopback.

## Development and verification

Install the full development profile before running all checks:

```bash
uv sync --locked --group dev --all-extras
make check
make smoke
```

| Command | Verification |
|---|---|
| `make lock-check` | Dependency declarations match the lockfile |
| `make fmt-check` / `make lint` | Formatting, imports, and lint |
| `make types` | Strict typing of package source |
| `make test` | Full test discovery; optional tests skip if extras are absent |
| `make test-full` | Full suite with required extras enforced |
| `make coverage` | Branch coverage measured locally |
| `make smoke` | Known-answer examples, save/load, reports, and dry-runs |
| `make release-check` | All checks, distribution build, Twine metadata checks, and isolated wheel installation |

Without Make, run `uv lock --check` and invoke the tools through `uv run --no-sync`; see [CONTRIBUTING.md](https://github.com/nehalvaghasiya/omni-eval/blob/main/CONTRIBUTING.md) for the full equivalents and hook setup.

`make setup-core` installs developer tools without optional extras. `make setup` installs the full profile. Full verification sets `OMNIEVAL_REQUIRE_EXTRAS=1` to fail if an extra is missing. Tests use saved/synthetic data and allow local loopback fixtures; they do not require real provider credentials.

CI covers Python 3.11–3.13 with core and full profiles, checks the actual interpreter, and verifies wheel installation. The release workflow verifies and uploads distributions; publication is a separate maintainer action.

For a separate build directory, use `make release-check DIST=/tmp/omni-eval-dist` with a fresh directory. To check an existing wheel, use `make smoke-wheel WHEEL=/path/to/package.whl`. Optional synthetic timing diagnostics are available with `uv run --no-sync python scripts/smoke.py --performance`; these are not representative model benchmarks.

## Troubleshooting

| Symptom | Action |
|---|---|
| Python version is unsupported | Use Python 3.11–3.13; pass `--python 3.12` to `uv sync` to select the repository default |
| `omni-eval` or a developer tool is missing | Run the appropriate `uv sync --locked` profile, then use `uv run --no-sync` |
| Parquet or the local API cannot import | Install `--extra parquet` or `--extra api`; repeat flags to keep both |
| The lockfile is out of date | After an intentional dependency change, run `uv lock` and review the diff; commit it with the dependency declaration |
| A score is unavailable or coverage is low | Inspect metric prerequisites, missing predictions, case errors, and reported denominators |
| A gate is inconclusive | Check metric availability and coverage; use `--strict` to surface this in CI |
| Integration tests cannot create local sockets | Run them in an environment permitting loopback fixtures |
| `init` refuses to create files | Choose a new workspace directory or edit the existing configuration |

## Project status and limitations

OmniEval 0.2.0 is a development preview. The supported core is native scoring of saved predictions with task presets, persisted evidence, reports, compatible-run comparisons, and threshold gates.

- Audio support evaluates transcripts; vision support evaluates supplied labels. Media inference is not implemented.
- RAG citation and context metrics assess supplied evidence; they do not establish factual faithfulness.
- Security metrics require explicit verified oracle results. Detector flags or generated claims are insufficient.
- Code and SQL checks do not execute predictions or prove functional correctness.
- External acquisition is serial, and complete cost accounting is not implemented.
- Upstream plugin integrations are unavailable until qualified. Synthetic demonstrations do not qualify a provider.
- The optional API lacks production authentication. Strong process isolation and DNS rebinding defenses require additional deployment work.

## Contributing and support

See [CONTRIBUTING.md](https://github.com/nehalvaghasiya/omni-eval/blob/main/CONTRIBUTING.md) for development setup, metric contributions, and pull-request checks.

Use [GitHub Issues](https://github.com/nehalvaghasiya/omni-eval/issues) for reproducible bugs and feature requests. For vulnerabilities, follow the private reporting guidance in [SECURITY.md](https://github.com/nehalvaghasiya/omni-eval/blob/main/SECURITY.md).

## License

OmniEval is distributed under the [MIT license](https://github.com/nehalvaghasiya/omni-eval/blob/main/LICENSE).
