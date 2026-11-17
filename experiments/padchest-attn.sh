#!/usr/bin/env bash
# PadChest-60 attention-head classifier: release labels, RAD-DINO patch tokens of every
# PadChest study stored as NVFP4, then the head trained on them (thresholds on the drawn
# PadChest val split).
set -euo pipefail
cd "$(dirname "$0")/.."

s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes "$@"; }

lab=$(s -- labrag labels --dataset padchest --labeler padchest60)
tok=$(s --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag embed --dataset padchest --feature tokens)
clf=$(s -d afterok:$lab:$tok --gres=gpu:1 --cpus-per-task=16 --mem=64G -- labrag classify --dataset padchest --labeler padchest60 --classifier attn --tokens nvfp4 --seed 1)
echo "labels $lab embed $tok classify $clf"
