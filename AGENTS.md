# AGENTS.md

Raycer is a standalone pseudo-3D racing game with column raycasting and an ML research
environment in the spirit of ViZDoom (Python 3.12, uv, numpy, CPU only, headless by default).
IDs such as `REND-06` or `API-21` in tests and docstrings name the requirement a test covers.

## Setup

```bash
uv sync --locked --all-extras --group dev
```

## Checks

The quality gate is `scripts/check.sh`; it must exit 0. It runs:

```bash
uv sync --locked --all-extras --group dev
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv run mypy
uv run pytest -m "not slow and not human" -q --cov=raycer --cov-fail-under=85
uv run raycer benchmark --scenario technical-v0 --steps 2000 \
  --buffers screen,depth,labels --seed 0 --min-sps "${RAYCER_MIN_SPS:-400}"
```

Always call tools with explicit paths (`src tests examples`), never on `.`.

## Conventions

- Package layout: `src/raycer/`, scenarios as package data
  in `src/raycer/scenarios/*.toml`, golden files in `tests/golden/*.npz`; regenerate them with
  `uv run pytest tests/test_golden.py --update-golden` and justify every golden change in the
  PR (QA-35).
- Headless: tests run with `SDL_VIDEODRIVER=dummy`. Never open windows.
- No `print` in `src/` except CLI output; use `logging`.
- Public functions have type hints and a one-line docstring. mypy strict on `src/raycer`.
- No new dependencies beyond those in `pyproject.toml`. PNGs are written without Pillow.
- Commit messages: English, imperative, subject at most 72 characters.
- PR body contains `Closes #<n>` and follows `.github/pull_request_template.md`.
