#!/usr/bin/env bash
set -euo pipefail
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYTHONHASHSEED=0 \
       OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1
step() { printf '\n== %s ==\n' "$1"; }
step sync;      uv sync --locked --all-extras --group dev
step lint;      uv run ruff check src tests examples
step format;    uv run ruff format --check src tests examples
step types;     uv run mypy
step tests;     uv run pytest -m "not slow and not human" -q \
                  --cov=raycer --cov-report=term-missing:skip-covered --cov-fail-under=85
echo "ALL CHECKS PASSED"
