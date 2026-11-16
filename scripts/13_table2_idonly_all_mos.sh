#!/usr/bin/env bash
# New experimental rows E1/E2 only. Does NOT rerun E3/E4.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
OUT="${TABLE2_OUTPUT:-results/table2_idonly_alpha07}"
DEVICE="${DEVICE:-cuda:0}"
MODEL="${CLIP_MODEL:-ViT-B/16}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
NEG_REPO="${NEG_REPO:-/data/hx/code/NegLabel}"

python -m table2_idonly.run \
  --dataset all --out "$OUT" --neglabel-root "$NEG_REPO" \
  --model "$MODEL" --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" --workers "$NUM_WORKERS" \
  --alpha 0.7 --prompt-idx 85 --max-samples "$MAX_SAMPLES" "$@"

if [[ "$MAX_SAMPLES" == 0 ]]; then
  python -m table2_idonly.report --input "$OUT" --output "$OUT" \
    --neglabel-root "$NEG_REPO" \
    --reference table2_idonly/reference_metrics_from_handoff.csv
else
  echo "[SMOKE ONLY] MAX_SAMPLES=$MAX_SAMPLES; no formal paper table created."
fi
