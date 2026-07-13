.PHONY: venv install dev seed demo test lint eval eval-cascade eval-three-phase burr-ui clean

# Portable across macOS/Linux: use python3 outside a venv; venv exposes `python`
PYTHON ?= python3

venv:
	$(PYTHON) -m venv .venv
	@echo "Now run:  source .venv/bin/activate"

install:
	$(PYTHON) -m pip install -e ".[dev]"

dev:
	uvicorn pantry_planner.api:app --reload --port 8000

seed:
	$(PYTHON) -m pantry_planner.db seed

demo:
	$(PYTHON) -m pantry_planner.demo pbj_sandwich

test:
	pytest -v

lint:
	ruff check pantry_planner/ tests/
	ruff format --check pantry_planner/ tests/

fmt:
	ruff format pantry_planner/ tests/

eval:
	$(PYTHON) -m evals.run

eval-cascade:
	ROUTING_STRATEGY=cascade $(PYTHON) -m evals.run

eval-three-phase:
	ROUTING_STRATEGY=three_phase $(PYTHON) -m evals.run

eval-compare:
	$(PYTHON) -m evals.router_eval

burr-ui:
	@echo "Opening Burr UI at http://localhost:7241"
	burr

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -f pantry.db burr_tracking.db
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
