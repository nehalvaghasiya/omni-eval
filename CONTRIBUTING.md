# Contributing to OmniEval

Thank you for helping improve OmniEval. Contributions should keep evaluation behavior explicit, reproducible, and supported by evidence.

## Development setup

Use Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/getting-started/installation/). The repository defaults to Python 3.12; CI uses uv 0.12.1.

```bash
git clone https://github.com/nehalvaghasiya/omni-eval.git
cd omni-eval
uv sync --locked --group dev --all-extras
uv run --no-sync omni-eval run -c examples/support-qa/candidate.yaml --out runs
```

This evaluates saved support answers and writes an HTML report. Follow the [README quick start](README.md#quick-start) to compare both prompt versions and inspect the failed answer.

For contributions that do not need optional dependencies, use `uv sync --locked --group dev`. API, Parquet, and independent statistics tests skip when their extras are absent; complete verification requires the full profile.

Install the optional Git hooks after preparing the environment:

```bash
uv run --no-sync pre-commit install
```

The hooks run the same Ruff installed from `uv.lock` and may fix lint or formatting issues. Review those changes before committing. To check hooks without installing them:

```bash
uv run --no-sync pre-commit run --all-files
```

## Repository layout

| Directory | Responsibility |
|---|---|
| `src/omni_eval/cli` | Command parsing and presentation |
| `src/omni_eval/core` | Evaluation, metric selection, comparisons, and gates |
| `src/omni_eval/metrics` | Native metrics and scoring evidence |
| `src/omni_eval/io`, `sources`, `storage` | Input parsing, prediction acquisition, and saved artifacts |
| `src/omni_eval/types` | Typed configuration, cases, scores, and results |
| `src/omni_eval/plugins` | Provider status and bounded worker/judge protocols |
| `src/omni_eval/reports`, `server` | Saved-run reports and the optional local API |
| `tests` | Unit, contract, integration, CLI, and security regression checks |
| `examples`, `scripts` | Saved-output evaluation workflows, task fixtures, and release smoke checks |

Keep business logic out of CLI modules and reuse the public evaluation engine. Follow existing module boundaries and naming conventions; contract tests enforce short source filenames and at most one top-level class per source module.

## Verify a change

With all extras installed:

```bash
make check
make smoke
```

`make check` checks lock consistency, formatting, lint, strict typing, the full suite, and CLI dry-runs. Equivalent commands without Make are:

```bash
uv lock --check
uv run --no-sync ruff check --select I .
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync mypy src/omni_eval
OMNIEVAL_REQUIRE_EXTRAS=1 uv run --no-sync pytest -q
uv run --no-sync pytest -q tests/e2e/test_demo.py
uv run --no-sync python scripts/smoke.py
```

For a focused iteration, select the relevant tests with `uv run --no-sync pytest -q PATH`. Full verification still runs before merge. To measure coverage, use `make coverage`.

Tests do not need real model credentials or external provider requests. HTTP/judge integration tests start loopback servers, so the test environment must permit local sockets. Keep fixtures synthetic and deterministic.

For packaging changes, use a fresh output directory:

```bash
make release-check DIST=/tmp/omni-eval-dist
```

This builds the source distribution and wheel, checks metadata with Twine, verifies examples, and installs the wheel with locked core dependencies in a temporary environment outside the checkout. Without Make, run the checks above, then:

```bash
uv build --out-dir /tmp/omni-eval-dist
uv run --no-sync python -m twine check /tmp/omni-eval-dist/*
uv run --no-sync python scripts/wheel.py --dist /tmp/omni-eval-dist
```

The release workflow verifies and uploads distributions. It does not publish to a package index.

## Dependency changes

Use `pyproject.toml` for dependency declarations and commit `uv.lock` alongside dependency changes.

```bash
uv add PACKAGE
uv add --dev TOOL
uv add --optional api PACKAGE
uv remove PACKAGE
```

Replace the example names with the intended dependency. Preserve optional extras for capabilities that are not needed by the core package. uv is a development tool, not a runtime dependency of OmniEval.

For an intentional update to an existing dependency:

```bash
uv lock --upgrade-package PACKAGE
uv sync --locked --group dev --all-extras
make check
```

Review version and transitive dependency changes. After editing declarations manually, run `uv lock`. Routine installation and CI use `--locked` so they cannot silently rewrite the lockfile. Dependabot proposes weekly uv and GitHub Actions updates; changes still require review.

## Adding or changing a metric

1. Define the formula, supported tasks, prerequisites, direction, unit, aggregation level, and undefined-input behavior.
2. Implement the `Metric` contract in the appropriate family. Use `Accuracy` as a small sample-level example; preserve the distinction between sample and corpus metrics.
3. Register the metric in the core registry. List only implemented capabilities.
4. Add independent known-answer cases, missing/error cases, coverage assertions, and relevant contract coverage. For a new native metric, extend the independent-answer mapping in the existing metric suite.
5. Add the registered name, prerequisites, and semantics to the README metric reference. Update the metric version when scoring semantics change.

Unavailable inputs must remain visible through status and coverage. Avoid inventing zeros, reference evidence, per-case corpus scores, or confidence intervals.

## Issues and pull requests

For bugs, include the installed OmniEval, Python, and uv versions, a minimal dataset/configuration, the exact command, expected behavior, and sanitized output. `uv run --no-sync omni-eval doctor` reports the package and Python versions.

For feature requests, describe the evaluation problem, a concrete example, and the expected result. Provider integrations need real qualification evidence before being described as supported.

Before opening a pull request:

- Explain the problem and resulting behavior.
- Run relevant checks and report their actual results.
- Update the README when commands, configuration, or metric behavior changes.
- Add a concise entry to the Unreleased changelog for user-visible changes.
- Review the diff for credentials, personal paths, and unrelated generated files.

Use [GitHub Issues](https://github.com/nehalvaghasiya/omni-eval/issues) for ordinary reports. Follow [the security policy](SECURITY.md) for vulnerabilities.

By contributing, you agree that your contributions are distributed under the repository's [MIT license](LICENSE).
