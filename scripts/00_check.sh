#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
echo "Project root: $PROJECT_ROOT"
echo "Official NegLabel root: $OFFICIAL_ROOT"
for file in mmcls/models/classifiers/multi_modal.py mmcls/models/ood/clip_ood.py custom_config/clip.py txtfiles/noun.act.txt txtfiles/adj.all.txt; do
  if [[ ! -e "$OFFICIAL_ROOT/$file" ]]; then
    echo "[WARN] Official file not found: $OFFICIAL_ROOT/$file"
  else
    echo "[PASS] $file"
  fi
done
python - <<'PY'
import importlib.util, torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available(): print('CUDA GPU:', torch.cuda.get_device_name(0))
for name in ['torchvision','mmcv','clip','numpy','sklearn','PIL','matplotlib','pytest']:
 print(name, 'OK' if importlib.util.find_spec(name) else 'MISSING')
PY
printf '\nFor the full run, use: bash scripts/01_compare.sh\n'
