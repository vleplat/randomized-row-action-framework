.PHONY: install install-dev test lint check quick figures-nnls figures-entropy figures-moffett clean

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e ".[dev,notebooks]"

test:
	python -m pytest

lint:
	python -m ruff check src experiments tests

check: test lint

quick:
	python -m pytest
	python -m experiments.simplex_nmf --mode synthetic --profile quick

figures-nnls:
	python -m experiments.noisy_nnls

figures-entropy:
	python -m experiments.entropy_selector

figures-moffett:
	python -m experiments.simplex_nmf --mode moffett --profile paper --data data/Moffet.mat

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info src/*.egg-info results figures/nnls_noisy figures/entropy_selector figures_moffett
