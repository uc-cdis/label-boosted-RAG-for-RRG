#!/usr/bin/env bash
# GRPO with thinking, importance-sampling control: the labrag core cell's train prompts
# (as grpo-smoke.sh; skipped once built), then 32 studies trained for 24 steps on one
# node's 4 GPUs with each of TRL's sequence_mask (its default) and token_truncate
# (configs/tuning.yaml grpo-thinking-is-*), to compare grad_norm and the IS ratios.
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
  j=$(s --cpus-per-task=8 --mem=96G -- labrag build-prompts --experiment train --dataset $dataset --section $JOINT --condition labrag --prompt $PSC --predictions $MI --devices --incidentals)
  ids+=" prompts.$dataset $j"
  dep+=":$j"
  prompts+=" train/$dataset/$JOINT/$MIL/$PSC"
done
for mode in sequence-mask token-truncate; do
  j=$(s -d afterok$dep --gres=gpu:4 --cpus-per-task=144 --mem=512G -- labrag tune --prompts$prompts --config grpo-thinking-is-$mode)
  ids+=" tune.$mode $j"
done
echo "$ids"
