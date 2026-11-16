#!/usr/bin/env bash
# Full 50k ImageNet-vs-4 MOS OOD, once each, frozen parameters.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU" LOCAL_RANK=0
python -m neglab_crop.dataset_setup verify \
  --project-root "$PROJECT_ROOT" --official-root "$OFFICIAL_ROOT" \
  --id-path "${ID_PATH:-/data/Public/imagenet/val}" --strict
OUT="${ATTENTION_OUTPUT:-results/mos_four_attention_alpha07}"
if [[ -e "$OUT/comparison.csv" ]]; then
  echo "[ERROR] Existing results $OUT/comparison.csv; set ATTENTION_OUTPUT to a new folder." >&2
  exit 2
fi
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
  --batch-size "${ATTENTION_BATCH_SIZE:-8}" \
  --view-batch-size "$VIEW_BATCH_SIZE" --workers "$WORKERS" \
  --dino-proposal-batch "${DINO_PROPOSAL_BATCH:-4}" \
  --alpha 0.7 --cls-mass 0.95 --cls-view mask \
  --include-fivecrop --check-official-parity \
  --output-dir "$OUT" \
  "$@" 2>&1 | tee "$OUT/run.log"
echo "[RESULT] $OUT/comparison.csv"
