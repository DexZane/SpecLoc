.PHONY: install install-cu118 install-cpu install-dev info doctor validate train evaluate test lint clean

PYTHON ?= python
DATASET ?= rsod

install:
	$(PYTHON) -m pip install -r requirements.txt

install-cu118:
	$(PYTHON) -m pip install -r requirements-cu118.txt

install-cpu:
	$(PYTHON) -m pip install -r requirements-cpu.txt

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

info:
	specloc info

doctor:
	specloc doctor $(DATASET)

validate:
	specloc validate $(DATASET)

train:
	specloc train $(DATASET)

evaluate:
	specloc evaluate $(DATASET)

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests scripts tools

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -prune -exec rm -rf {} +
	find . -type d -name '.ruff_cache' -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
