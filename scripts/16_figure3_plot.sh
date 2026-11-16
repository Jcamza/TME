#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SCORES_DIR="${SCORES_DIR:-results/mos_four_alpha07/scores}"
FIGURE3_OUTPUT="${FIGURE3_OUTPUT:-results/figure3_analysis}"
BOOTSTRAP="${BOOTSTRAP:-2000}"
SEED="${SEED:-20261112}"
# Optional: INPUT_CONFIG=figure3_inputs.json for CSV with arbitrary filenames/columns.
ARGS=()
if [[ -n "${INPUT_CONFIG:-}" ]]; then
  ARGS+=(--config "$INPUT_CONFIG")
else
  ARGS+=(--scores-dir "$SCORES_DIR")
fi
python -m figure3.run "${ARGS[@]}" \
 --out "$FIGURE3_OUTPUT" \
 --bootstrap "$BOOTSTRAP" --seed "$SEED" \
 --neglabel-root "${NEGLABEL_ROOT:-/data/hx/code/NegLabel}" "$@"
