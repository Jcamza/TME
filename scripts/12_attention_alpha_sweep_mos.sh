#!/usr/bin/env bash
# One-pass alpha-grid study for CLS-95 mask and Object-DINO BBox.
# No change to official NegLabel, no retuning of FiveCrop.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU" LOCAL_RANK=0

python -m neglab_crop.dataset_setup verify \
  --project-root "$PROJECT_ROOT" --official-root "$OFFICIAL_ROOT" \
  --id-path "${ID_PATH:-/data/Public/imagenet/val}" --strict

OUT="${ATTENTION_ALPHA_OUTPUT:-results/mos_four_attention_alpha_sweep_cls95}"
if [[ -e "$OUT/comparison.csv" ]]; then
  echo "[ERROR] $OUT/comparison.csv exists; set ATTENTION_ALPHA_OUTPUT to a fresh directory." >&2
  exit 2
fi
mkdir -p "$OUT"

# CLS attention and DINO regions/CLIP embeddings are each extracted ONE TIME per image.
# The alpha grid is evaluated inside the same forward pass. Existing FiveCrop
# reference at alpha=0.7 is available under results/mos_four_alpha07.
python -m neglab_crop.attention_alpha_runner \
  --official-root "$OFFICIAL_ROOT" \
  --id-path "${ID_PATH:-/data/Public/imagenet/val}" \
  --id-ann "${ID_ANN:-$PROJECT_ROOT/data/imagenet_val_ood_index.txt}" \
  --ood-override "iNaturalist=$OFFICIAL_ROOT/data/ood_data/iNaturalist/images" \
  --ood-override "SUN=$OFFICIAL_ROOT/data/ood_data/SUN/images" \
  --ood-override "Places=$OFFICIAL_ROOT/data/ood_data/Places/images" \
  --ood-override "Textures=$OFFICIAL_ROOT/data/ood_data/Textures/dtd/images_collate" \
  --dataset iNaturalist --dataset SUN --dataset Places --dataset Textures \
  --dino-model "${DINO_MODEL:-/data/Public/models/dinov2-base}" \
  --batch-size "${ATTENTION_BATCH_SIZE:-8}" \
  --view-batch-size "$VIEW_BATCH_SIZE" --workers "$WORKERS" \
  --dino-proposal-batch "${DINO_PROPOSAL_BATCH:-4}" \
  --alpha 0.7 \
  --alpha-grid "${ATTENTION_ALPHA_GRID:-0,0.05,0.1,0.2,0.3,0.5,0.7,1.0}" \
  --cls-mass 0.95 --cls-view mask \
  --check-official-parity --visualize-n 0 --no-plot \
  --output-dir "$OUT" \
  "$@" 2>&1 | tee "$OUT/run.log"

python -m neglab_crop.alpha_sweep_report \
  --comparison "$OUT/comparison.csv" \
  --output-dir "$OUT"
echo "[RESULT] $OUT/alpha_sweep_summary.csv"
