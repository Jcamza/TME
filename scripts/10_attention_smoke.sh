#!/usr/bin/env bash
# One tiny trial over ID + all four MOS OOD datasets. Never use partial metrics as evidence.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU" LOCAL_RANK=0
python -m neglab_crop.dataset_setup verify \
  --project-root "$PROJECT_ROOT" --official-root "$OFFICIAL_ROOT" \
  --id-path "${ID_PATH:-/data/Public/imagenet/val}" --strict
OUT="${ATTENTION_SMOKE_OUTPUT:-results/attention_smoke}"
mkdir -p "$OUT"
set -o pipefail
python -m neglab_crop.attention_runner \
  --official-root "$OFFICIAL_ROOT" \
  --id-path "${ID_PATH:-/data/Public/imagenet/val}" \
  --id-ann "$PROJECT_ROOT/data/imagenet_val_ood_index.txt" \
  --ood-override "iNaturalist=$OFFICIAL_ROOT/data/ood_data/iNaturalist/images" \
  --ood-override "SUN=$OFFICIAL_ROOT/data/ood_data/SUN/images" \
  --ood-override "Places=$OFFICIAL_ROOT/data/ood_data/Places/images" \
  --ood-override "Textures=$OFFICIAL_ROOT/data/ood_data/Textures/dtd/images_collate" \
  --dataset iNaturalist --dataset SUN --dataset Places --dataset Textures \
  --dino-model "${DINO_MODEL:-facebook/dinov2-base}" \
  --batch-size 4 --view-batch-size 16 --workers 2 \
  --dino-proposal-batch 2 --alpha 0.7 --cls-mass 0.9 \
  --cls-view mask --include-fivecrop --check-official-parity \
  --limit-id 16 --limit-ood 16 --output-dir "$OUT" \
  "$@" 2>&1 | tee "$OUT/run.log"
echo "[RESULT] $OUT/comparison.csv"
