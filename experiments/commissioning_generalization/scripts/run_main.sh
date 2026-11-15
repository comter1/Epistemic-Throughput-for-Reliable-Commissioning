#!/usr/bin/env bash
set -euo pipefail
: "${CGX_DATA_ROOT:?Set CGX_DATA_ROOT to the qualified external artifact root}"
: "${CGX_PYTHON:?Set CGX_PYTHON to the isolated environment Python executable}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export CGX_REMOTE_WORKERS_FILE="${CGX_REMOTE_WORKERS_FILE:-$CGX_DATA_ROOT/reports/remote_workers.json}"
export CGX_EVAL_WORKERS="${CGX_EVAL_WORKERS:-16}" CGX_FEATURE_READERS="${CGX_FEATURE_READERS:-4}"
exec "$CGX_PYTHON" -m cgx.cli run-main --data-root "$CGX_DATA_ROOT" --config "$CGX_DATA_ROOT/software/protocol.json" --gpus "${CGX_GPUS:-0,1}" --workers "${CGX_WORKERS:-16}" --resume
