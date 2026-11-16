#!/usr/bin/env bash
# Inventory without modifying sources or downloading anything.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
cd "$PROJECT_ROOT"
exec python -m neglab_crop.dataset_setup check \
  --project-root "$PROJECT_ROOT" --official-root "$OFFICIAL_ROOT" \
  --id-path "${ID_PATH:-/data/Public/imagenet/val}" "$@"
