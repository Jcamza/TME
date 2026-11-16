#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SCORES_DIR="${SCORES_DIR:-results/mos_four_alpha07/scores}"
python -m figure3.run --scores-dir "$SCORES_DIR" --inspect
