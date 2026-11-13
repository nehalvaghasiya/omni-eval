.DEFAULT_GOAL := help

UV ?= uv
RUN = $(UV) run --no-sync

.PHONY: help setup format format-check lint check typecheck test

help:
	@printf '%s\n' \
	  'make setup         Install the project and development dependencies' \
	  'make format        Sort imports and format Python files with Ruff' \
	  'make format-check  Check imports and formatting without changing files' \
	  'make lint          Run all configured Ruff lint rules' \
	  'make check         Check formatting and lint' \
	  'make typecheck     Run mypy on the package' \
	  'make test          Run the test suite'

setup:
	$(UV) sync --group dev

format:
	$(RUN) ruff check --select I --fix .
	$(RUN) ruff format .

format-check:
	$(RUN) ruff check --select I .
	$(RUN) ruff format --check .

lint:
	$(RUN) ruff check .

check: format-check lint

typecheck:
	$(RUN) mypy src/omni_eval

test:
	$(RUN) pytest -q
