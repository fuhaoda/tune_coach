#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing virtualenv Python at $PYTHON_BIN"
  echo "Create it with:"
  echo "  python -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  pip install -U pip"
  echo "  pip install -e ."
  exit 1
fi

cd "$ROOT_DIR"
exec "$PYTHON_BIN" -m tune_coach
