#!/usr/bin/env bash
# Run extraction from project root: bash run_extraction.sh [--limit N] [--resume]
set -e
cd "$(dirname "$0")"
[[ -f .env ]] && set -a && source .env && set +a

PY=""
if [[ -x venv/bin/python ]]; then
  PY=venv/bin/python
elif python3 -c "import anthropic" 2>/dev/null; then
  PY=python3
fi
if [[ -z "$PY" ]]; then
  echo "No Python with anthropic found. Install deps first:"
  echo "  python3 -m pip install -r requirements.txt"
  echo "Or create a venv: python3 -m venv venv && venv/bin/pip install -r requirements.txt"
  exit 1
fi
exec "$PY" extract.py "$@"
