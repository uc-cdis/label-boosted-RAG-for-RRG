#!/usr/bin/env bash
# Device types predicted from the image, on the prior-report base, on all of MIMIC-CXR
# and ReXGradient val (Findings+Impression, 2 sampling seeds):
# - the new base: the prior report in every prompt, instruction 4 narrowed
#   (joint-*-prior-changes-*: compare finding by finding where the image supports it,
#   no general "unchanged"), for retrieval-only and the multiclass pooled fine-tune,
#   vs prior-details-val's joint-*-prior-* cells;
# - the pooled fine-tune with the device types as extra labels (finetune --labeler
#   chexbert14.devices --multiclass): its labels with the predicted device types
#   (shown and ranked on, as oracle.devices) and without them (CheXbert-14 only: the
#   control for retraining), and the oracle with device types on the new base.
# Qwen3.5-27B with the image (518 px). Labels, embeds, the CheXbert-14 fine-tune and
# its predicts are skipped, as are prior-details-val's cells; new are CheXpert Plus's
# device labels, the device fine-tune and its predicts, and the 5 new cells.
set -euo pipefail
cd "$(dirname "$0")/.."

s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes "$@"; }
SET=val
MODEL=qwen3.5-27b-vision
SAMPLING="qwen3.5-thinking-32k qwen3.5-thinking-32k-seed2"
SEC=findings-impression
JOINT=findings+impression
POOL=chexpertplus-mimic-rexgradient
MC=$POOL/raddino.ft-attn-seed1/$SEC/chexbert14.mc
MD=$POOL/raddino.ft-attn-seed1/$SEC/chexbert14.devices.mc
PS=joint-simple-negatives-indication-image
PNP=joint-naive-negatives-indication-prior-image
PSP=joint-simple-negatives-indication-prior-image
PNC=joint-naive-negatives-indication-prior-changes-image
PSC=joint-simple-negatives-indication-prior-changes-image

labels=""
declare -A lab dev
for dataset in chexpertplus mimic rexgradient; do
  for section in findings impression; do
    lab[$dataset.$section]=$(s --gres=gpu:1 -- labrag labels --dataset $dataset --section $section)
    dev[$dataset.$section]=$(s --cpus-per-task=8 -- labrag labels --dataset $dataset --section $section --labeler devices)
    labels+=":${lab[$dataset.$section]}:${dev[$dataset.$section]}"
  done
done
mc=$(s -d afterok$labels --gres=gpu:4 --cpus-per-task=144 --mem=256G -- labrag finetune --dataset $POOL --section $SEC --seed 1 --multiclass)
md=$(s -d afterok$labels --gres=gpu:4 --cpus-per-task=144 --mem=256G -- labrag finetune --dataset $POOL --section $SEC --seed 1 --multiclass --labeler chexbert14.devices)
ids="labels ${labels#:} finetune $mc $md"

gens() { for p in $2; do echo -n "$p/$MODEL.$1/$SET "; done; }
# pool <name> <prompt dir>: NAME=run,run over the sampling seeds
pool() {
  local r=""
  for sampling in $SAMPLING; do r+=" $(gens $sampling "$2")"; done
  echo -n "$1=$(echo $r | tr ' ' ',') "
}
for dataset in mimic rexgradient; do
  F=val/$dataset/$JOINT
  B="labrag build-prompts --experiment val --dataset $dataset --section $JOINT"
  lj=$(s -d afterok:${lab[$dataset.findings]}:${lab[$dataset.impression]} --cpus-per-task=8 -- labrag labels --dataset $dataset --section $JOINT)
  dj=$(s -d afterok:${dev[$dataset.findings]}:${dev[$dataset.impression]} --cpus-per-task=8 -- labrag labels --dataset $dataset --section $JOINT --labeler devices)
  emb=$(s --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag embed --dataset $dataset)
  prm=$(s -d afterok:$mc --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag predict --dataset $dataset --classifier $MC)
  prd=$(s -d afterok:$md --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag predict --dataset $dataset --classifier $MD)
  ids+=" $dataset: labels $lj devices $dj embed $emb predict $prm $prd"
  RO=$F/retrieval-only
  MCL=$F/labrag.$POOL.ft-attn-seed1.$SEC.chexbert14.mc
  MDL=$F/labrag.$POOL.ft-attn-seed1.$SEC.chexbert14.devices.mc
  ORD=$F/labrag-oracle.chexbert14.devices
  dep=""
  # cell <after> <prompt dir> <build-prompts args>: prompts, then generate + evaluate
  # per sampling seed
  cell() {
    local after=$1 dir=$2
    shift 2
    local pro g e
    pro=$(s -d afterok:$after --cpus-per-task=8 -- $B "$@")
    ids+=" prompts $pro"
    for sampling in $SAMPLING; do
      g=$(s -d afterok:$pro --gres=gpu:1 -- labrag generate --prompts $dir --set $SET --model $MODEL --sampling $sampling)
      e=$(s -d afterok:$g --gres=gpu:1 -- labrag evaluate --generations $(gens $sampling $dir))
      ids+=" $g $e"
      dep+=":$e"
    done
  }
  # prior-details-val's cells (built: skipped), then the new ones
  cell $lj:$emb $RO/$PNP --condition retrieval-only --prompt $PNP
  cell $lj:$prm:$emb $MCL/$PSP --condition labrag --predictions $MC --prompt $PSP
  cell $lj:$dj:$emb $ORD/$PS --condition labrag-oracle --prompt $PS --devices
  cell $lj:$emb $RO/$PNC --condition retrieval-only --prompt $PNC
  cell $lj:$prm:$emb $MCL/$PSC --condition labrag --predictions $MC --prompt $PSC
  cell $lj:$prd:$emb $MDL/$PSC --condition labrag --predictions $MD --prompt $PSC
  cell $lj:$dj:$prd:$emb $MDL.devices/$PSC --condition labrag --predictions $MD --prompt $PSC --devices
  cell $lj:$dj:$emb $ORD/$PSC --condition labrag-oracle --prompt $PSC --devices
  cells="$(pool retrieval-only.negatives.indication.prior $RO/$PNP)"
  cells+="$(pool mc.positives.negatives.indication.prior $MCL/$PSP)"
  cells+="$(pool oracle.negatives.indication.devices $ORD/$PS)"
  cells+="$(pool retrieval-only.negatives.indication.prior-changes $RO/$PNC)"
  cells+="$(pool mc.positives.negatives.indication.prior-changes $MCL/$PSC)"
  cells+="$(pool mc-devices.positives.negatives.indication.prior-changes $MDL/$PSC)"
  cells+="$(pool mc-devices.positives.negatives.indication.prior-changes.devices $MDL.devices/$PSC)"
  cells+="$(pool oracle.negatives.indication.prior-changes.devices $ORD/$PSC)"
  cmp=$(s -d afterok$dep -- labrag compare --generations $cells --out $F/qwen3.5-27b.devices/$SET)
  ids+=" compare $cmp"
done
echo "$ids"
