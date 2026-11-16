#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$PROJECT_ROOT/config.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/config.env"
  set +a
else
  echo "[INFO] config.env absent: using defaults. cp config.env.example config.env to customize."
fi
export OFFICIAL_ROOT="${OFFICIAL_ROOT:-/data/hx/code/NegLabel}"
export GPU="${GPU:-0}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export VIEW_BATCH_SIZE="${VIEW_BATCH_SIZE:-32}"
export WORKERS="${WORKERS:-2}"
export ALPHA_GRID="${ALPHA_GRID:-0.2}"
export FUSION="${FUSION:-feature}"
export CROP_MODE="${CROP_MODE:-fivecrop}"
export OUTPUT_DIR="${OUTPUT_DIR:-results/neglab_compare_fivecrop}"
