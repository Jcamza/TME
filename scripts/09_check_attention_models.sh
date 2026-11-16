#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
cd "$PROJECT_ROOT"
echo '===== Existing NegLabel and data ====='
python -m neglab_crop.dataset_setup verify \
  --project-root "$PROJECT_ROOT" --official-root "$OFFICIAL_ROOT" \
  --id-path "${ID_PATH:-/data/Public/imagenet/val}" --strict

echo '===== DINOv2 model availability ====='
DINO_MODEL="${DINO_MODEL:-facebook/dinov2-base}"
python - "$DINO_MODEL" <<'PY'
from pathlib import Path
import sys
mid=sys.argv[1]
try:
    import transformers
    print('[PASS] transformers',transformers.__version__)
except Exception as e:
    print('[MISSING] transformers:',repr(e))
    sys.exit(2)
try:
    import sklearn
    print('[PASS] sklearn',sklearn.__version__)
except Exception as e:
    print('[MISSING] sklearn:',repr(e))
    sys.exit(2)
p=Path(mid).expanduser()
if p.exists():
    print('[MODEL] local model directory:',p.resolve())
    assert (p/'config.json').is_file(),f'Missing {p}/config.json'
    assert any((p/f).is_file() for f in ('model.safetensors','pytorch_model.bin')), 'Missing model weights in local model directory'
else:
    print('[MODEL] HuggingFace repo ID:',mid)
    print('NOTE: the full runner defaults to local cache ONLY; it will never download by accident.')
    print('Pass --allow-model-download if downloads are intended, or set DINO_MODEL to a local HF directory.')
PY

echo '===== Candidate local DINO paths ====='
find /data/Public/models -maxdepth 4 \( -iname '*dinov2*' -o -iname '*dino_v2*' \) -print 2>/dev/null | head -30 || true
printf '\nTo specify a local HF model, add to config.env:\nDINO_MODEL=/data/Public/models/dinov2-base\n'
