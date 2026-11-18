#!/usr/bin/env bash
# Classifier-only gate for the pooled end-to-end classifier (no generation): RAD-DINO
# fine-tuned with the attention head (LP-FT), Findings + Impression readouts, seed 1,
# trained on CheXpert Plus alone and on CheXpert Plus + MIMIC-CXR + ReXGradient. Each
# is then predicted on all three datasets with thresholds on that dataset's own val.
# Read out against the frozen sections head on CheXpert Plus val: wrong labels per
# study at its miss rate. Each fine-tune takes a whole node (image decoding is the
# bottleneck). Labels already built are skipped. kg35-nvl02 is left out until its data
# sync is confirmed.
set -euo pipefail
cd "$(dirname "$0")/.."

s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes -x kg35-nvl02 "$@"; }
SEC=findings-impression

labels=""
for dataset in chexpertplus mimic rexgradient; do
  for section in findings impression; do
    labels+=":$(s --gres=gpu:1 -- labrag labels --dataset $dataset --section $section)"
  done
done

for pool in chexpertplus chexpertplus-mimic-rexgradient; do
  ft=$(s -d afterok$labels --gres=gpu:4 --cpus-per-task=144 --mem=256G -- labrag finetune --dataset $pool --section $SEC --seed 1)
  for target in chexpertplus mimic rexgradient; do
    s -d afterok:$ft --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag predict --dataset $target --classifier $pool/raddino.ft-attn-seed1/$SEC/chexbert14
  done
done
