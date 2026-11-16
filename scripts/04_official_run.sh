#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
export CUDA_VISIBLE_DEVICES="$GPU"
export LOCAL_RANK=0
cd "$OFFICIAL_ROOT"
exec bash ./run.sh
