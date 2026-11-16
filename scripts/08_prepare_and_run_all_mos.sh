#!/usr/bin/env bash
# Example: bash scripts/08_prepare_and_run_all_mos.sh --download-missing
set -euo pipefail
source "$(dirname "$0")/_common.sh"
cd "$PROJECT_ROOT"
python -m neglab_crop.dataset_setup check \
  --project-root "$PROJECT_ROOT" --official-root "$OFFICIAL_ROOT" \
  --id-path "${ID_PATH:-/data/Public/imagenet/val}"
bash scripts/06_prepare_mos_datasets.sh "$@"
bash scripts/07_run_all_mos.sh
