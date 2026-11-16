#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
cd "$PROJECT_ROOT"
python -m pytest tests -q
printf '\nUnit tests PASSED. For a real GPU/data smoke test:\n'
printf 'bash scripts/01_compare.sh --limit-id 32 --limit-ood 32 --dataset iNaturalist\n'
