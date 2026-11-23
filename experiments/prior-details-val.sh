#!/usr/bin/env bash
# Context the label list lacks, on all of MIMIC-CXR and ReXGradient val
# (Findings+Impression, 2 sampling seeds), each cell next to its joint-val cell:
# - retrieval-only and the multiclass pooled fine-tune's positives with the report of
#   the patient's previous study in the prompt (joint-*-prior-*: "None" without one;
#   comparing with it allowed), vs joint-val's same cells without it;
# - the oracle with device types in the label set (labels --labeler devices: shown and
#   ranked on), and the oracle with each label's size or severity (severity stage: shown
#   only), vs joint-val's oracle: ceilings, before training a classifier for either.
# Qwen3.5-27B with the image (518 px). Labels, embeds, the fine-tune and predicts that
# earlier experiments built are skipped; new are the device labels, the severities and
# the 4 cells.
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
PN=joint-naive-negatives-indication-image
PS=joint-simple-negatives-indication-image
PNP=joint-naive-negatives-indication-prior-image
PSP=joint-simple-negatives-indication-prior-image

labels=""
declare -A lab
for dataset in chexpertplus mimic rexgradient; do
  for section in findings impression; do
    lab[$dataset.$section]=$(s --gres=gpu:1 -- labrag labels --dataset $dataset --section $section)
    labels+=":${lab[$dataset.$section]}"
  done
done
mc=$(s -d afterok$labels --gres=gpu:4 --cpus-per-task=144 --mem=256G -- labrag finetune --dataset $POOL --section $SEC --seed 1 --multiclass)
ids="labels ${labels#:} finetune $mc"

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
  df=$(s --cpus-per-task=8 -- labrag labels --dataset $dataset --section findings --labeler devices)
  di=$(s --cpus-per-task=8 -- labrag labels --dataset $dataset --section impression --labeler devices)
  dj=$(s -d afterok:$df:$di --cpus-per-task=8 -- labrag labels --dataset $dataset --section $JOINT --labeler devices)
  sev=$(s -d afterok:$lj --cpus-per-task=8 -- labrag severity --dataset $dataset --section $JOINT)
  emb=$(s --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag embed --dataset $dataset)
  prm=$(s -d afterok:$mc --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag predict --dataset $dataset --classifier $MC)
  ids+=" $dataset: labels $lj devices $df $di $dj severity $sev embed $emb predict $prm"
  RO=$F/retrieval-only
  MCL=$F/labrag.$POOL.ft-attn-seed1.$SEC.chexbert14.mc
  ORD=$F/labrag-oracle.chexbert14
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
  # joint-val's cells (built: skipped), then the new ones
  cell $lj:$emb $RO/$PN --condition retrieval-only --prompt $PN
  cell $lj:$prm:$emb $MCL/$PS --condition labrag --predictions $MC --prompt $PS
  cell $lj:$emb $ORD/$PS --condition labrag-oracle --prompt $PS
  cell $lj:$emb $RO/$PNP --condition retrieval-only --prompt $PNP
  cell $lj:$prm:$emb $MCL/$PSP --condition labrag --predictions $MC --prompt $PSP
  cell $lj:$dj:$emb $ORD.devices/$PS --condition labrag-oracle --prompt $PS --devices
  cell $lj:$sev:$emb $ORD.severity/$PS --condition labrag-oracle --prompt $PS --severity
  cells="$(pool retrieval-only.negatives.indication $RO/$PN)"
  cells+="$(pool mc.positives.negatives.indication $MCL/$PS)"
  cells+="$(pool oracle.negatives.indication $ORD/$PS)"
  cells+="$(pool retrieval-only.negatives.indication.prior $RO/$PNP)"
  cells+="$(pool mc.positives.negatives.indication.prior $MCL/$PSP)"
  cells+="$(pool oracle.negatives.indication.devices $ORD.devices/$PS)"
  cells+="$(pool oracle.negatives.indication.severity $ORD.severity/$PS)"
  cmp=$(s -d afterok$dep -- labrag compare --generations $cells --out $F/qwen3.5-27b.prior-details/$SET)
  ids+=" compare $cmp"
done
echo "$ids"
