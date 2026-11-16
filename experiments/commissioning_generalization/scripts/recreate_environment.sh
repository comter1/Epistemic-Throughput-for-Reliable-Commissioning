#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/recreate_environment.sh /absolute/new/venv" >&2
  exit 2
fi
CGX_EXPERIMENT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
python3 -m venv "$1"
"$1/bin/python" -m pip install --upgrade pip
"$1/bin/python" -m pip install -r "$CGX_EXPERIMENT_DIR/requirements.lock.txt" --extra-index-url https://download.pytorch.org/whl/cu130
"$1/bin/python" -m pip install --no-deps -e "$CGX_EXPERIMENT_DIR"
