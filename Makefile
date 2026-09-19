.PHONY: install dev lint format typecheck test test-all test-lookahead cov build docker-up clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

lint:
	ruff check quantlab tests examples benchmarks

format:
	ruff check --fix quantlab tests examples benchmarks
	ruff format quantlab tests examples benchmarks

typecheck:
	mypy quantlab

test:
	pytest

test-all:
	pytest

test-lookahead:
	pytest -m lookahead

cov:
	pytest --cov=quantlab --cov-report=term-missing

build:
	python -m build

docker-up:
	docker compose up --build

clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache build dist *.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} +