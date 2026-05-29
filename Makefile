.PHONY: test lint format demo calibrate

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

demo:
	@echo "demo not implemented yet — see Phase 5"
	@exit 1

calibrate:
	@echo "calibrate not implemented yet — see Phase 4"
	@exit 1
