#!/usr/bin/env bash
# GRPO with thinking, smoke run: the labrag core cell's train prompts (CheXpert Plus,
# MIMIC-CXR, ReXGradient; the deployed 40-label fine-tune's in-sample predictions,
# chexpertplus-train-labels.sh first), then the tuning/ images built and 8 studies
# trained for 2 steps on one node's 4 GPUs (configs/tuning.yaml grpo-thinking-smoke).
# The prompt artifacts are the pilot's and the full run's (skipped there).
set -euo pipefail
cd "$(dirname "$0")/.."

s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes "$@"; }
JOINT=findings+impression
POOL=chexpertplus-mimic-rexgradient
MI=$POOL/raddino.ft-attn-seed1/findings-impression/chexbert14.devices.incidentals.mc
MIL=labrag.$POOL.ft-attn-seed1.findings-impression.chexbert14.devices.incidentals.mc.devices.incidentals
PSC=joint-simple-negatives-indication-prior-changes-image

ids=""
dep=""
prompts=""
for dataset in chexpertplus mimic rexgradient; do
  # ~40 min and ~51 GB for MIMIC-CXR's 121k targets
  j=$(s --cpus-per-task=8 --mem=96G -- labrag build-prompts --experiment train --dataset $dataset --section $JOINT --condition labrag --prompt $PSC --predictions $MI --devices --incidentals)
  ids+=" prompts.$dataset $j"
  dep+=":$j"
  prompts+=" train/$dataset/$JOINT/$MIL/$PSC"
done
j=$(s -d afterok$dep --gres=gpu:4 --cpus-per-task=144 --mem=512G -- labrag tune --prompts$prompts --config grpo-thinking-smoke)
ids+=" tune $j"
echo "$ids"
