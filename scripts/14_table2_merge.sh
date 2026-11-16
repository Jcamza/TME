#!/usr/bin/env bash
# Rebuild summary and LaTeX without running the CLIP encoder.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
OUT="${TABLE2_OUTPUT:-results/table2_idonly_alpha07}"
NEG_REPO="${NEG_REPO:-/data/hx/code/NegLabel}"
REFERENCE="${REFERENCE_CSV:-table2_idonly/reference_metrics_from_handoff.csv}"
python -m table2_idonly.report --input "$OUT" --output "$OUT" \
  --neglabel-root "$NEG_REPO" --reference "$REFERENCE" "$@"
