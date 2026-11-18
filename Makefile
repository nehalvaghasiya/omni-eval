.DEFAULT_GOAL := help
UV := uv
RUN := $(UV) run --no-sync
TASK ?= classification
PLUGIN ?= fixture
WHEEL ?=
DIST ?= dist
.PHONY: help setup setup-core lock lock-check fmt fmt-check lint types test test-full test-unit test-int test-e2e test-sec coverage dry dry-all check build smoke smoke-wheel clean plugin-setup plugin-test release-check
help:
	@echo 'setup (all extras) | setup-core | lock | fmt | lint | types | test | test-full | dry-all | check | build | smoke'
setup:
	$(UV) sync --locked --group dev --all-extras
setup-core:
	$(UV) sync --locked --group dev
lock:
	$(UV) lock
lock-check:
	$(UV) lock --check
fmt:
	$(RUN) ruff check --select I --fix .
	$(RUN) ruff format .
fmt-check:
	$(RUN) ruff check --select I .
	$(RUN) ruff format --check .
lint:
	$(RUN) ruff check .
types:
	$(RUN) mypy src/omni_eval
test:
	$(RUN) pytest -q
test-full:
	OMNIEVAL_REQUIRE_EXTRAS=1 $(RUN) pytest -q
test-unit:
	$(RUN) pytest -q tests/unit tests/contracts
test-int:
	$(RUN) pytest -q tests/integration
test-e2e:
	$(RUN) pytest -q tests/e2e
test-sec:
	$(RUN) pytest -q tests/security
coverage:
	OMNIEVAL_REQUIRE_EXTRAS=1 $(RUN) pytest --cov=omni_eval --cov-branch --cov-report=term-missing
dry:
	$(RUN) omni-eval demo $(TASK) --dry-run
dry-all:
	$(RUN) pytest -q tests/e2e/test_demo.py
check: lock-check fmt-check lint types test-full dry-all
build:
	$(UV) build --out-dir "$(DIST)"
smoke:
	$(RUN) python scripts/smoke.py
smoke-wheel:
	$(RUN) python scripts/wheel.py $(if $(WHEEL),--wheel "$(WHEEL)",--dist "$(DIST)")
plugin-setup:
	$(RUN) omni-eval plugins setup $(PLUGIN)
plugin-test:
	$(RUN) pytest -q tests/integration/test_plugin.py
release-check: check
	$(MAKE) build
	$(RUN) python -m twine check "$(DIST)"/*
	$(MAKE) smoke
	$(MAKE) smoke-wheel
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
