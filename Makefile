PYTHON ?= python

.PHONY: install test lint format check cli

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

test-fast:
	$(PYTHON) -m pytest -m "not slow"

lint:
	ruff check .

format:
	ruff check --fix .
	ruff format .

check: lint
	ruff format --check .
	$(PYTHON) -m pytest

cli:
	quantscope --help
