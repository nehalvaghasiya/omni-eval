# OmniEval

A local-first, typed Python package for **evaluating saved AI predictions**,
comparing runs, and optionally acquiring predictions from an explicitly allowed
HTTP model endpoint. The same canonical records feed native classification, text,
retrieval, RAG, agent, and latency metrics.

> **Status: implementation snapshot.** This is not a production-certified or
> PyPI-published release.

[Quick start](#quick-start) · [Python API](#python-api) · [CLI reference](#cli-reference) · [Input contract](#input-contract) · [Endpoints and judges](#endpoints-and-judges) · [Development](#development)

## Quick start

Requires **Python 3.11–3.13** and [uv](https://docs.astral.sh/uv/). From the
repository root:

```bash
uv sync --locked --group dev

# Evaluate the included saved predictions
uv run --no-sync omni-eval evaluate --config suites/basic.yaml
uv run --no-sync omni-eval evaluate --config suites/rag.yaml
uv run --no-sync omni-eval evaluate --config suites/agent.yaml

# Try a synthetic evaluation without contacting a model endpoint
uv run --no-sync omni-eval evaluate --dry-run

# Run the test suite
uv run --no-sync pytest -q
```

The example suites cover [classification](suites/basic.yaml),
[RAG](suites/rag.yaml), and [agents](suites/agent.yaml). Saved runs are written
under `.omni-eval/runs/` by default.

The repository includes `uv.lock`; use `--locked` to install its verified dependency
versions. Installation on a clean machine requires access to a Python package
index. Run `uv lock` when intentionally updating dependency requirements.

Alternatively, install the existing wheel locally once its dependencies are
available:

```bash
pip install dist/omni_eval-0.1.0-py3-none-any.whl
```

## Python API

Evaluate saved records, inspect metric coverage, and reload the stored run:

```python
from omni_eval import compare, evaluate, load_run

run = evaluate(
    samples="examples/classification/samples.jsonl",
    predictions="examples/classification/predictions.jsonl",
    metrics=["accuracy", "macro_f1", "brier_score", "ece"],
    output_dir=".omni-eval/runs",
)

print([(metric.name, metric.value, metric.coverage) for metric in run.metrics])

restored = load_run(run.path)

# With another compatible saved run:
# comparisons = compare(other_run, restored, paired=True)
```

Paired comparisons use shared case IDs and require retained sample and prediction
records. Metric prerequisites, coverage, and statuses remain part of the results.

## CLI reference

Use `uv run --no-sync` before these commands when working from the repository, or
run them directly after installing the package. Replace paths in angle brackets
with your own files or saved run folders.

| Task | Command |
| --- | --- |
| Show help | `omni-eval --help` |
| List native metrics | `omni-eval metrics list` |
| Inspect metric prerequisites | `omni-eval metrics inspect accuracy` |
| Preview project initialization | `omni-eval init --dry-run` |
| Validate saved records | `omni-eval validate --samples <samples.jsonl> --predictions <predictions.jsonl>` |
| Evaluate a suite | `omni-eval evaluate --config suites/basic.yaml` |
| Compare saved runs | `omni-eval compare --baseline <baseline-run> --candidate <candidate-run>` |
| Re-score a saved run | `omni-eval replay --run <run-dir>` |
| Render a saved report | `omni-eval report --run <run-dir> --format html` |
| Check local capabilities | `omni-eval doctor --dry-run` |
| Judge a saved run | `omni-eval judge --run <run-dir> --config examples/endpoint/judge.yaml` |
| Import completed external runner results | `omni-eval runner --result <result.json>` |
| Explore saved runs locally | `omni-eval serve --runs .omni-eval/runs` |

**Every operational command supports a synthetic, network-free `--dry-run`.**
For example:

```bash
uv run --no-sync omni-eval judge --dry-run
uv run --no-sync omni-eval runner --dry-run
uv run --no-sync omni-eval serve --dry-run
```

Dry runs are clearly marked as synthetic. The live read-only server requires the
optional API dependencies, installed with
`uv sync --locked --group dev --extra api`.

## Input contract

Inputs use Pydantic v2 models that reject unknown top-level fields:

| Model | Required fields | Common optional fields |
| --- | --- | --- |
| [`EvalSample`](src/omni_eval/schemas/eval_sample.py) | `sample_id`, `input` | `reference`, `metadata`, `extensions` |
| [`PredictionRecord`](src/omni_eval/schemas/prediction_record.py) | `sample_id`, `output` | `probabilities`, `contexts`, `trace`, `rag_trace`, `timing`, `error`, `extensions` |

JSONL files contain **one record per line**. Records join by `sample_id`
regardless of order. Prediction probabilities use an explicit class label order:

**Sample** — one line in a sample JSONL file:

```json
{"sample_id": "s-001", "input": {"text": "Example input"}, "reference": {"label": "yes"}}
```

**Prediction** — the matching line in a prediction JSONL file:

```json
{"sample_id": "s-001", "output": {"label": "yes"}, "probabilities": {"labels": ["no", "yes"], "values": [0.1, 0.9], "kind": "softmax"}}
```

Missing predictions and reference prerequisites affect validation, coverage, and
statuses; they are **never silently scored as zero**. Offline timing is explicitly
labelled `measured_external`; endpoint timing is labelled `measured_by_omnieval`.

CSV ingestion requires explicit field mapping through the Python API. See the
[included examples](examples/) for classification, text, retrieval, RAG, agent,
and audio records.

### Stored runs and record retention

Stored runs are immutable local folders with a manifest and artifact hash checks.
They include aggregate and sample-level scores, errors, and the original sample
and prediction records by default. HTML reports escape displayed content;
Markdown and JUnit reports are also available through suite configuration.

Set the following in a suite when raw records must not be persisted:

```yaml
report:
  formats: [json, html]
  retain_records: false
```

This omits the raw sample and prediction files. **Replay and paired comparison
from that artifact require those records and will therefore be unavailable.**

## Endpoints and judges

### HTTP prediction sources

Use [`examples/endpoint/endpoint.yaml`](examples/endpoint/endpoint.yaml) as a
**development-only example**. Live endpoints require explicit URL configuration
and allowed hosts. The client:

- Rejects private and metadata IP ranges, with an explicit exception for approved
  loopback development.
- Disables redirects and environment proxies.
- Bounds request timeouts, response sizes, retries, and concurrency.

External data-use approval is the caller's responsibility.
**Application-level DNS pre-resolution is not a substitute for a network egress
firewall** in an untrusted server deployment.

### Judge evaluation

The `judge` command requires a saved run and explicit rubric/provider
configuration, or `--dry-run` for a clearly synthetic fixture. See
[`examples/endpoint/judge.yaml`](examples/endpoint/judge.yaml).

| Built-in rubric ID | Criterion |
| --- | --- |
| `answer-correctness-v1` | Answer correctness relative to a supplied reference |
| `groundedness-v1` | Factual support in the supplied retrieved contexts |
| `answer-relevance-v1` | Relevance to the input question |

**External judge scores are estimates, not ground-truth oracle verdicts.**



## Development

Use the Makefile for the repository's development commands:

| Command | Purpose |
| --- | --- |
| `make setup` | Install the project and development dependencies |
| `make format` | Sort imports, then format all Python files with Ruff |
| `make format-check` | Check import order and formatting without editing files |
| `make lint` | Run all configured Ruff lint rules |
| `make check` | Run formatting and lint checks |
| `make typecheck` | Run mypy on the package |
| `make test` | Run the test suite |
| `make help` | List available targets |

Run `make setup` again after development dependencies change. Other targets use
those installed dependencies without syncing the environment.

The development group includes the optional API dependencies and JSON Schema
stubs so strict type checks and API tests cover the full package. Core wheel
installations keep those API dependencies optional. Mypy follows the active
Python interpreter; CI checks Python 3.11, 3.12, and 3.13 with the same lockfile.

Ruff uses an 88-character line length, four-space indentation, double quotes, and
separate standard library, third-party, and project import groups. Its version
matches the pre-commit hooks. Automatic fixes are restricted to imports so
formatting preserves program logic.

Keep one blank line between logical sections inside functions. Ruff preserves
these separators but does not infer or insert them automatically. Trailing commas
keep longer expressions expanded across lines when needed.

## License

[MIT](LICENSE).
