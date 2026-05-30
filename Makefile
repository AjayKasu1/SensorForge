.PHONY: test lint format coverage demo calibrate

test:
	uv run pytest

# Enforce the >80% rule from the constitution on the correctness-critical
# packages. Scoped to isp/ and metrics/ so unbuilt phases don't skew it.
coverage:
	uv run pytest --cov=sensorforge.isp --cov=sensorforge.metrics \
		--cov-report=term-missing --cov-fail-under=80

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

demo:
	@echo "demo not implemented yet; see Phase 5"
	@exit 1

calibrate:
	@echo "calibrate not implemented yet; see Phase 4"
	@exit 1
