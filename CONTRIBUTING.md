# Contributing to SensorForge

Thanks for your interest. This is a focused project; small, well-scoped PRs are
easiest to review.

## Setup

```bash
uv sync          # install everything, including dev tools
make test        # run the suite
make lint        # ruff check + format check
make coverage    # enforce >80% on isp/ and metrics/
```

Python 3.11. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

## Conventions

- **Commits**: conventional style (`feat:`, `fix:`, `docs:`, `refactor:`,
  `test:`, `chore:`), one logical change per commit.
- **Style**: `ruff` is the linter and formatter (line length 100). Run
  `make format` before committing.
- **Tests**: add tests for new logic; the suite must stay green and coverage on
  `isp/` and `metrics/` above 80%.
- **Decisions**: anything architectural gets a short ADR in `docs/adr/` (see the
  existing ones for the format).
- **Physics first**: parameters carry units; physical models cite a paper,
  datasheet, or standard.

## Scope

See [LIMITATIONS.md](LIMITATIONS.md) for what is intentionally out of scope.
Geometric registration and full ColorChecker/CCM calibration are the main
open areas (v2); contributions there are especially welcome.
