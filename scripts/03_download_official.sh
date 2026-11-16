#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
if [[ -e "$OFFICIAL_ROOT/.git" || -e "$OFFICIAL_ROOT/mmcls" ]]; then
  echo "Already present: $OFFICIAL_ROOT. No changes made."
  exit 0
fi
mkdir -p "$(dirname "$OFFICIAL_ROOT")"
echo "Cloning original official code (requires network): https://github.com/tmlr-group/NegLabel"
git clone https://github.com/tmlr-group/NegLabel.git "$OFFICIAL_ROOT"
echo "Original NegLabel checkout created at: $OFFICIAL_ROOT"
echo "Install official requirements following its README; this script does NOT modify official source."
