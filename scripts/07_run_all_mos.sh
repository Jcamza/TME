#!/usr/bin/env bash
# One model initialization, one ID pass, 4 OOD passes; no parameter search.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
cd "$PROJECT_ROOT"
python -m neglab_crop.dataset_setup verify \
  --project-root "$PROJECT_ROOT" --official-root "$OFFICIAL_ROOT" \
  --id-path "${ID_PATH:-/data/Public/imagenet/val}" --strict

RUN_OUTPUT_DIR="${MOS_RUN_OUTPUT_DIR:-results/mos_four_alpha07}"
if [[ -e "$RUN_OUTPUT_DIR/comparison.csv" ]]; then
  echo "[ERROR] Existing results at $RUN_OUTPUT_DIR/comparison.csv; set MOS_RUN_OUTPUT_DIR to a new path." >&2
  exit 2
fi
mkdir -p "$RUN_OUTPUT_DIR"
LOG_FILE="$RUN_OUTPUT_DIR/run.log"
set -o pipefail
bash scripts/01_compare.sh \
  --id-path "${ID_PATH:-/data/Public/imagenet/val}" \
  --id-ann "$PROJECT_ROOT/data/imagenet_val_ood_index.txt" \
  --ood-override "iNaturalist=$OFFICIAL_ROOT/data/ood_data/iNaturalist/images" \
  --ood-override "SUN=$OFFICIAL_ROOT/data/ood_data/SUN/images" \
  --ood-override "Places=$OFFICIAL_ROOT/data/ood_data/Places/images" \
  --ood-override "Textures=$OFFICIAL_ROOT/data/ood_data/Textures/dtd/images_collate" \
  --dataset iNaturalist --dataset SUN --dataset Places --dataset Textures \
  --fusion feature --alpha-grid 0.7 --output-dir "$RUN_OUTPUT_DIR" \
  "$@" 2>&1 | tee "$LOG_FILE"
echo "[RESULT] $RUN_OUTPUT_DIR/comparison.csv"
