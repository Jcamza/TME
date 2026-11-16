#!/usr/bin/env bash
# Symlink-only data views in NegLabel/data/ood_data, except explicit downloads if requested.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
cd "$PROJECT_ROOT"
exec python -m neglab_crop.dataset_setup prepare \
  --project-root "$PROJECT_ROOT" --official-root "$OFFICIAL_ROOT" \
  --id-path "${ID_PATH:-/data/Public/imagenet/val}" "$@"
