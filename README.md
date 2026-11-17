# OmniEval

One file, one command, transparent scores. Evaluate saved predictions, classification, text, retrieval, RAG, agent/MCP traces, transcripts, and synthetic security cases locally.

> Development preview 0.2.0. The implemented offline workflows pass the local verification matrix on Python 3.11–3.13. Twenty upstream integrations are still planned and unavailable. See [verification evidence and limitations](IMPLEMENTATION.md).

## Start here

Use Python 3.11, 3.12, or 3.13. For a minimal editable installation:

```bash
python -m pip install -e .
omni-eval demo classification
omni-eval demo rag
omni-eval init my-eval --task classification
omni-eval run -c my-eval/eval.yaml --out runs
```

The demos use synthetic cases and run real native scoring without model keys or downloads. `init` creates a runnable configuration for any of the 11 tasks: `classification`, `text`, `retrieval`, `rag`, `agent`, `mcp`, `audio`, `vision`, `security`, `code`, and `sql`.

Optional installation profiles:

| Install | Enables |
|---|---|
| `python -m pip install -e '.[api]'` | Local read-only FastAPI run browser |
| `python -m pip install -e '.[parquet]'` | Parquet input |
| `python -m pip install -e '.[stats]'` | Independent NumPy/scikit-learn reference checks; native metrics work without them |
| `make setup` | Locked developer tools and all three extras |
| `make setup-core` | Locked developer tools and core dependencies only |

Package installation can require network access. Ordinary evaluations and tests use saved/synthetic data; HTTP and judge verification uses local fake services.

## Evaluate your own data

Save `cases.jsonl` with one JSON object per line:

```jsonl
{"id":"1","input":"Refund?","expected":"30 days","actual":"30 days"}
{"id":"2","input":"Shipping?","expected":"2 days","actual":"3 days"}
```

Run it directly:

```bash
omni-eval run cases.jsonl --task text --metric exact --metric token_f1 --out runs
```

Or save `eval.yaml` next to the data:

```yaml
version: 1
task: text
data: cases.jsonl
metrics: [exact, token_f1]
```

```bash
omni-eval run -c eval.yaml --out runs
```

Paths in YAML resolve relative to the YAML file. JSONL is the default format; CSV requires explicit column mapping, and Parquet requires its extra. Split predictions join by case ID, never by row position. Duplicated/unknown IDs and ambiguous inline-plus-split predictions are rejected. Missing predictions remain visible in coverage.

The same engine is available in Python:

```python
from omni_eval import evaluate, load, compare, metrics

run = evaluate("cases.jsonl", task="text", metrics=["exact", "token_f1"], out="runs")
print(run.summary)
print(run.metric("exact").value)  # 0.5 for the two cases above
print(run.failures(metric="exact"))
restored = load(run.path)
print(compare(run, restored, seed=42).summary)
print(metrics(task="text"))
```

`evaluate` also accepts a typed `Config` or iterable of `Case` objects. An empty dataset is rejected consistently for files and in-memory inputs.

## Choose metrics

`omni-eval metrics list` lists 64 registered names, including aliases and k templates. Use a concrete name such as `recall@5` or `topk@2` in configurations. `omni-eval metrics show recall@5` shows tasks, prerequisites, version, unit, and direction.

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

## Understand results and compare runs

Every score reports `value`, `status`, `n_total`, `n_eligible`, `n_scored`, `n_failed`, `n_missing`, and coverage (`n_scored / n_total`). A high score on a small eligible subset does not establish full-dataset performance. Undefined or failed scores use null, with status/reason evidence. A corpus score need not have a per-case scalar.

Saved runs contain configuration, provenance, metric versions, checksums, and optional case/point/error records. `output: {retain: false}` suppresses those records; replay then requires another retained source. Loading rejects altered, added, missing, incomplete, or symlinked artifacts.

```bash
omni-eval report runs/BASELINE_ID --format html
omni-eval report runs/BASELINE_ID --format json
omni-eval report runs/BASELINE_ID --format md
omni-eval report runs/BASELINE_ID --format junit
omni-eval compare runs/BASELINE_ID runs/CANDIDATE_ID --metric accuracy
omni-eval check runs/CANDIDATE_ID -c gates.yaml --strict
```

Replace the run IDs with actual saved directory names. A minimal `gates.yaml` is:

```yaml
gates:
  - metric: accuracy
    op: gte
    value: 0.8
    min_coverage: 1.0
```

Comparison pairs stable IDs and checks inputs, references, metric settings, versions, units, sources, and latency methodology. Bootstrap intervals are deterministic for a fixed seed and require paired sample scores; corpus-only metrics do not receive fabricated intervals. Slices use metadata keys and suppress small groups (default minimum support 5).

Gates return pass, fail, or inconclusive. CLI exits are 0 for normal completion, 1 for a failed threshold, 2 for invalid input, 3 for runtime/provider failure, 4 for an inconclusive gate with `--strict`, and 5 for policy refusal. Without `--strict`, inconclusive outcomes are displayed with exit 0; use strict mode in CI. `min_pairs` on an absolute saved-run gate is inconclusive because pairing evidence is unavailable there.

Native scoring policy is version 2 after correctness fixes; saved older runs remain readable, but comparison rejects differing metric versions. Synthetic results are identified in human reports, JSON, and JUnit.

## Local transports and optional judge

`source: http` and `source: openai` require an explicit target, allowlisted hosts/ports, and environment-referenced endpoint/token. Both honor configured retries and record measured timing. Requests are currently serial; concurrent acquisition and full cost accounting remain future work. Test fixtures exercise HTTP failures without contacting real providers.

`judge:relevance` is an estimated rubric score from an explicitly configured OpenAI-compatible endpoint. Configure `model`, `rubric`, `url_env`, `allowed_hosts`, finite call/token budgets, and `approve_external_data: true`; see `examples/judge/rubric.yaml`. The fake local judge verifies the protocol, consent, evidence, errors, and budgets. Actual model quality remains unverified.

`run --dry-run` substitutes synthetic inputs and the fixture worker for external evaluation; it does not send the dataset to a provider. `fixture:echo` always marks its result synthetic. `plugins list/show/doctor/setup` expose capability status; planned workers refuse real execution. There are no qualified Ragas, DeepEval, scanner, GPU, or benchmark integrations yet.

The optional API is created with `omni_eval.server.app.create_app(runs)`. It exposes read-only health, metric, run-list, and run-summary routes. Bind it only to loopback. HTTPS/allowlist checks and redirect refusal are tested, but production authentication, strong sandboxing, and DNS rebinding defenses still require deployment work.

## Reproduce verification

```bash
make setup
make check
make smoke
make coverage
make release-check
uv run --no-sync python scripts/smoke.py --performance
```

`make check` requires all extras and runs lock consistency, formatting, lint, strict typing, the complete suite, and command dry-runs. `make release-check` additionally builds distributions, checks metadata with Twine, and installs the wheel plus locked core dependencies in a fresh environment outside the checkout. To keep artifacts separate, set `DIST=/tmp/omni-eval-dist`; an existing wheel can be selected with `make smoke-wheel WHEEL=/absolute/path/to/package.whl`.

`make setup-core && make test` verifies the minimal profile. API, Parquet, and independent statistics checks skip explicitly when their extras are absent. Full verification sets `OMNIEVAL_REQUIRE_EXTRAS=1` and fails immediately if an extra is missing. Tests permit loopback connections only.

Current evidence: all 64 metric names have independent known-answer assertions; all 11 task examples pass; the full suite passes on Python 3.11–3.13. Coverage and performance measurements, exact commands, remaining limitations, and roadmap status are maintained in [IMPLEMENTATION.md](IMPLEMENTATION.md). [PLAN.md](PLAN.md) and [INTEGRATIONS_PLAN.md](INTEGRATIONS_PLAN.md) describe future work, not completed capabilities.
