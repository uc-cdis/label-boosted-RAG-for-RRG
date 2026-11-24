#!/usr/bin/env bash
# Incidental findings as extra labels, oracle ceiling, on all of MIMIC-CXR and
# ReXGradient val (Findings+Impression, 2 sampling seeds): the reference incidental
# findings (labels --labeler incidentals: degenerative spine, aortic calcification, low
# lung volumes, ...) join the oracle's CheXbert-14 + device-type label set, shown and
# ranked on, on the prior-changes base (oracle.devices.incidentals vs devices-val's
# oracle.devices). Run after devices-val: its labels, fine-tunes, predicts and cells
# are skipped here (retrieval-only, mc-devices with device types, oracle.devices); new
# are the incidental labels and one cell per dataset.
set -euo pipefail
cd "$(dirname "$0")/.."

s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes -x kg35-nvl02 "$@"; }
SET=val
MODEL=qwen3.5-27b-vision
SAMPLING="qwen3.5-thinking-32k qwen3.5-thinking-32k-seed2"
SEC=findings-impression
JOINT=findings+impression
POOL=chexpertplus-mimic-rexgradient
MD=$POOL/raddino.ft-attn-seed1/$SEC/chexbert14.devices.mc
PNC=joint-naive-negatives-indication-prior-changes-image
PSC=joint-simple-negatives-indication-prior-changes-image

labels=""
declare -A lab dev inc
for dataset in chexpertplus mimic rexgradient; do
  for section in findings impression; do
    lab[$dataset.$section]=$(s --gres=gpu:1 -- labrag labels --dataset $dataset --section $section)
    dev[$dataset.$section]=$(s --cpus-per-task=8 -- labrag labels --dataset $dataset --section $section --labeler devices)
    labels+=":${lab[$dataset.$section]}:${dev[$dataset.$section]}"
    if [ $dataset != chexpertplus ]; then
      inc[$dataset.$section]=$(s --cpus-per-task=8 -- labrag labels --dataset $dataset --section $section --labeler incidentals)
    fi
  done
done
md=$(s -d afterok$labels --gres=gpu:4 --cpus-per-task=144 --mem=256G -- labrag finetune --dataset $POOL --section $SEC --seed 1 --multiclass --labeler chexbert14.devices)
ids="labels ${labels#:} finetune $md"

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
  ij=$(s -d afterok:${inc[$dataset.findings]}:${inc[$dataset.impression]} --cpus-per-task=8 -- labrag labels --dataset $dataset --section $JOINT --labeler incidentals)
  emb=$(s --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag embed --dataset $dataset)
  prd=$(s -d afterok:$md --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag predict --dataset $dataset --classifier $MD)
  ids+=" $dataset: labels $lj devices $dj incidentals ${inc[$dataset.findings]} ${inc[$dataset.impression]} $ij embed $emb predict $prd"
  RO=$F/retrieval-only
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
  # devices-val's cells (built: skipped), then the new one
  cell $lj:$emb $RO/$PNC --condition retrieval-only --prompt $PNC
  cell $lj:$dj:$prd:$emb $MDL.devices/$PSC --condition labrag --predictions $MD --prompt $PSC --devices
  cell $lj:$dj:$emb $ORD/$PSC --condition labrag-oracle --prompt $PSC --devices
  cell $lj:$dj:$ij:$emb $ORD.incidentals/$PSC --condition labrag-oracle --prompt $PSC --devices --incidentals
  cells="$(pool retrieval-only.negatives.indication.prior-changes $RO/$PNC)"
  cells+="$(pool mc-devices.positives.negatives.indication.prior-changes.devices $MDL.devices/$PSC)"
  cells+="$(pool oracle.negatives.indication.prior-changes.devices $ORD/$PSC)"
  cells+="$(pool oracle.negatives.indication.prior-changes.devices.incidentals $ORD.incidentals/$PSC)"
  cmp=$(s -d afterok$dep -- labrag compare --generations $cells --out $F/qwen3.5-27b.incidentals/$SET)
  ids+=" compare $cmp"
done
echo "$ids"
