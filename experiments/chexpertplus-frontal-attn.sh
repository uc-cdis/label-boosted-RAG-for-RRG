#!/usr/bin/env bash
# Full CheXpert Plus attention head: RAD-DINO patch tokens of every study with a frontal
# image (NVFP4), then one head with Findings and Impression readouts sharing each
# label's query, trained on every labelled train study, 3 seeds. All on one node, so
# the three trainings read the ~110 GB store from its page cache, not 3 times over NFS.
set -euo pipefail
cd "$(dirname "$0")/.."

NODE=kg35-nvl01
s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes -w $NODE "$@"; }

tok=$(s --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag embed --dataset chexpertplus --feature tokens)
clf=""
for seed in 1 2 3; do
  clf+=" $(s -d afterok:$tok --gres=gpu:1 --cpus-per-task=16 --mem=64G -- labrag classify --dataset chexpertplus --classifier attn --tokens nvfp4 --seed $seed)"
done
echo "embed $tok classify$clf"
