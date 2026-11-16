#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export LOCAL_RANK=0
cmd=(python -m neglab_crop.runner
  --official-root "$OFFICIAL_ROOT"
  --config custom_config/clip.py
  --output-dir "$OUTPUT_DIR"
  --fusion "$FUSION"
  --crop-mode "$CROP_MODE"
  --alpha-grid "$ALPHA_GRID"
  --batch-size "$BATCH_SIZE"
  --view-batch-size "$VIEW_BATCH_SIZE"
  --workers "$WORKERS"
  --check-official-parity)
[[ -n "${CLIP_WEIGHTS:-}" ]] && cmd+=(--clip-weights "$CLIP_WEIGHTS")
[[ -n "${ID_PATH:-}" ]] && cmd+=(--id-path "$ID_PATH")
[[ -n "${ID_ANN:-}" ]] && cmd+=(--id-ann "$ID_ANN")
[[ -n "${OOD_INATURALIST:-}" ]] && cmd+=(--ood-override "iNaturalist=$OOD_INATURALIST")
[[ -n "${OOD_SUN:-}" ]] && cmd+=(--ood-override "SUN=$OOD_SUN")
[[ -n "${OOD_PLACES:-}" ]] && cmd+=(--ood-override "Places=$OOD_PLACES")
[[ -n "${OOD_TEXTURES:-}" ]] && cmd+=(--ood-override "Textures=$OOD_TEXTURES")
cmd+=("$@")
echo "[RUN] ${cmd[*]}"
"${cmd[@]}"
