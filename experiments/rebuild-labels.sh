#!/usr/bin/env bash
# The device-type and incidental-finding labels rebuilt with the frozen rules (labels
# config rules_sha256), then what reads them: the Findings+Impression labels of
# MIMIC-CXR and ReXGradient, the pooled 40-label fine-tune (CheXbert-14 + device types
# + incidental findings, multiclass; epoch picked on every dataset's full val) and its
# predicts. Every job overwrites (the old artifacts are refused as built from other
# rules or stale), so submit once; core-test.sh then skips all of it.
set -euo pipefail
cd "$(dirname "$0")/.."

s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes "$@"; }
SEC=findings-impression
JOINT=findings+impression
POOL=chexpertplus-mimic-rexgradient
MI=$POOL/raddino.ft-attn-seed1/$SEC/chexbert14.devices.incidentals.mc

ids=""
deps=""
declare -A rule
for dataset in chexpertplus mimic rexgradient; do
  for labeler in devices incidentals; do
    rule[$dataset.$labeler]=""
    for section in findings impression; do
      j=$(s --cpus-per-task=8 -- labrag labels --dataset $dataset --section $section --labeler $labeler --overwrite)
      rule[$dataset.$labeler]+=":$j"
      deps+=":$j"
      ids+=" $dataset.$section.$labeler $j"
    done
  done
done
for dataset in mimic rexgradient; do
  for labeler in devices incidentals; do
    j=$(s -d afterok${rule[$dataset.$labeler]} --cpus-per-task=8 -- labrag labels --dataset $dataset --section $JOINT --labeler $labeler --overwrite)
    ids+=" $dataset.joint.$labeler $j"
  done
done
ft=$(s -d afterok$deps --gres=gpu:4 --cpus-per-task=144 --mem=256G -- labrag finetune --dataset $POOL --section $SEC --seed 1 --multiclass --labeler chexbert14.devices.incidentals --overwrite)
ids+=" finetune $ft"
for dataset in mimic rexgradient; do
  j=$(s -d afterok:$ft --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag predict --dataset $dataset --classifier $MI --overwrite)
  ids+=" predict.$dataset $j"
done
echo "$ids"
