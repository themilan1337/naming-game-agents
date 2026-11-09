#!/usr/bin/env bash
# One-command verification in a temporary copy: the original results/ is never touched.
#   bash scripts/verify_in_copy.sh           tests, ruff, full 3000-run table-s8, table gate
#   bash scripts/verify_in_copy.sh --full    also fetch_data, empirical, tipping, topologies, charts
# Needs uv. Works on Linux and macOS. Prints the seconds of every step.
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
if [ -f "$SRC/.verify_copy_marker" ]; then
  echo "verify_in_copy: refusing to run inside a copy of itself" >&2
  exit 2
fi
FULL=0
[ "${1:-}" = "--full" ] && FULL=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

TMP="$(mktemp -d "${TMPDIR:-/tmp}/yng_verify.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/yang-naming-game"
( cd "$SRC" && tar --exclude=.venv --exclude=data/cache --exclude=__pycache__ \
    --exclude=.pytest_cache --exclude=.ruff_cache -cf - . ) | ( cd "$TMP/yang-naming-game" && tar -xf - )
touch "$TMP/yang-naming-game/.verify_copy_marker"
cd "$TMP/yang-naming-game"
echo "verify_in_copy: working in $TMP/yang-naming-game (deleted at the end)"

step() {
  local name="$1"; shift
  local t0=$SECONDS
  echo "--- $name"
  "$@"
  echo "--- $name: $((SECONDS - t0)) s"
}

step "uv sync --locked" uv sync --locked
step "pytest" uv run pytest -q
step "ruff check" uv run ruff check .
step "ruff format --check" uv run ruff format --check .
step "table-s8 (3000 runs x 15 settings)" uv run python -m namegame.cli table-s8
step "check_table_s8" uv run python scripts/check_table_s8.py results/repro_tableS8_convergence.csv
if [ "$FULL" = 1 ]; then
  step "fetch_data" bash scripts/fetch_data.sh
  step "empirical" uv run python -m namegame.cli empirical
  step "tipping" uv run python -m namegame.cli tipping
  step "topologies" uv run python -m namegame.cli topologies
  step "charts" uv run python -m namegame.cli charts
fi
echo "verify_in_copy: all checks passed"
