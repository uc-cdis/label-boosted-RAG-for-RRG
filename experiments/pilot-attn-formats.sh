#!/usr/bin/env bash
# Token storage formats for the attention-head classifier: fp16 tokens embedded once,
# then the head trained on them read as fp16, fp8 and NVFP4, 3 seeds each (val AUROC).
# All on one node: embed stages ~100 GB in its /tmp, and the 9 trainings then read the
# tokens file from that node's page cache instead of 9 times over NFS.
set -euo pipefail
cd "$(dirname "$0")/.."

NODE=kg35-nvl01
s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes -w $NODE "$@"; }

tok=$(s --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag embed --dataset chexpertplus --feature tokens --section findings)
clf=""
for fmt in fp16 fp8 nvfp4; do
  for seed in 1 2 3; do
    clf+=" $(s -d afterok:$tok --gres=gpu:1 --cpus-per-task=16 --mem=64G -- labrag classify --dataset chexpertplus --section findings --classifier attn --tokens $fmt --seed $seed)"
  done
done
echo "embed $tok classify$clf"
