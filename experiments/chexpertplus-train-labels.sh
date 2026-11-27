#!/usr/bin/env bash
# CheXpert Plus into the GRPO training pool: its Findings+Impression reference labels
# (CheXbert-14, device types, incidental findings; the per-section labels are
# rebuild-labels.sh's) and the deployed pooled 40-label fine-tune's predictions for
# every study, train role included, as MIMIC-CXR and ReXGradient already have.
set -euo pipefail
cd "$(dirname "$0")/.."

s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes "$@"; }
POOL=chexpertplus-mimic-rexgradient
MI=$POOL/raddino.ft-attn-seed1/findings-impression/chexbert14.devices.incidentals.mc

ids=""
for labeler in chexbert14 devices incidentals; do
  j=$(s --cpus-per-task=8 -- labrag labels --dataset chexpertplus --section findings+impression --labeler $labeler)
  ids+=" joint.$labeler $j"
done
j=$(s --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag predict --dataset chexpertplus --classifier $MI)
ids+=" predict $j"
echo "$ids"
