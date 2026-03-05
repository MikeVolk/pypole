#* Variables
SHELL := /usr/bin/env bash
PYTHON := python
PYTHONPATH := `pwd`

#* Installation
.PHONY: install
install:
	uv sync

.PHONY: pre-commit-install
pre-commit-install:
	uv run pre-commit install

#* Formatters
.PHONY: codestyle
codestyle:
	uv run ruff format .
	uv run ruff check --fix .

.PHONY: formatting
formatting: codestyle

#* Linting
.PHONY: test
test:
	PYTHONPATH=$(PYTHONPATH) uv run pytest -c pyproject.toml --cov-report=html --cov=pypole tests/
	uv run coverage-badge -o assets/images/coverage.svg -f

.PHONY: check-codestyle
check-codestyle:
	uv run ruff format --check .
	uv run ruff check .

.PHONY: ty
ty:
	uv run ty check pypole

.PHONY: lint
lint: test check-codestyle ty

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: dsstore-remove
dsstore-remove:
	find . | grep -E ".DS_Store" | xargs rm -rf

.PHONY: mypycache-remove
mypycache-remove:
	find . | grep -E ".mypy_cache" | xargs rm -rf

.PHONY: ipynbcheckpoints-remove
ipynbcheckpoints-remove:
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf

.PHONY: pytestcache-remove
pytestcache-remove:
	find . | grep -E ".pytest_cache" | xargs rm -rf

.PHONY: cleanup
cleanup: pycache-remove dsstore-remove mypycache-remove ipynbcheckpoints-remove pytestcache-remove

.PHONY: pre-commit
pre-commit:
	uv run pre-commit run --all-files
