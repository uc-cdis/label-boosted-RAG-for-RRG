#!/usr/bin/env bash
# Literature baselines on TEST, Findings+Impression, MIMIC-CXR and ReXGradient: each
# published model run as published (its upstream's data prep, inputs and decoding; its
# own container, baselines/), on the core experiment's targets, scored with our suite.
# A yardstick for labrag - retrieval-only, not a generator of ours. Compared with the
# core experiment's retrieval-only and labrag pools (core-test.sh: run that first).
#   cxrmate-2   CXRMate-2 (Nicolson et al. 2026); parity with its published test
#               reports is in the generations' manifest
set -euo pipefail
cd "$(dirname "$0")/.."

s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes "$@"; }
SET=test
JOINT=findings+impression
MODEL=qwen3.5-27b-vision
SAMPLING="qwen3.5-thinking-32k qwen3.5-thinking-32k-seed2 qwen3.5-thinking-32k-seed3"
MI=labrag.chexpertplus-mimic-rexgradient.ft-attn-seed1.findings-impression.chexbert14.devices.incidentals.mc.devices.incidentals
PNC=joint-naive-negatives-indication-prior-changes-image
PSC=joint-simple-negatives-indication-prior-changes-image

# pool <name> <prompt dir>: NAME=run,run,run over the sampling seeds
pool() {
  local r=""
  for sampling in $SAMPLING; do r+="$2/$MODEL.$sampling/$SET,"; done
  echo -n "$1=${r%,} "
}
ids=""
for dataset in mimic rexgradient; do
  F=$SET/$dataset/$JOINT
  dep=""
  cells="$(pool retrieval-only $F/retrieval-only/$PNC)$(pool labrag $F/$MI/$PSC)"
  for model in cxrmate-2; do
    # 4 GPUs: one container each, on a quarter of the targets
    b=$(s --gres=gpu:4 --cpus-per-task=64 --mem=128G -- labrag baseline --model $model --dataset $dataset --set $SET)
    e=$(s -d afterok:$b --gres=gpu:1 -- labrag evaluate --generations $F/baseline/$model/$SET)
    ids+=" $dataset $model $b $e"
    dep+=":$e"
    cells+="$model=$F/baseline/$model/$SET "
  done
  cmp=$(s -d afterok$dep -- labrag compare --generations $cells --out $F/baselines/$SET)
  ids+=" compare $cmp"
done
echo "$ids"
