#!/usr/bin/env bash
# The core experiment (B) on TEST, Findings+Impression, MIMIC-CXR and ReXGradient,
# 3 sampling seeds per cell: Qwen3.5-27B with the image (518 px). Every condition has
# the same base inputs: the image, the indication, the prior report and the
# prior-changes instructions. Labels come from the pooled 40-label fine-tune
# (rebuild-labels.sh: submit that first and let it finish; here it is skipped, and
# labels from other rules are refused).
#   no-context      base inputs only: no examples, no labels
#   retrieval-only  examples by image similarity
#   filter-only     examples by label overlap; no labels shown
#   labels-only     labels shown; no examples
#   labrag          LaB-RAG: label-ranked examples and labels shown, all 40 labels
#   labrag-shuffled labrag with another patient's predicted labels
#   oracle          labrag with the target's reference labels (the ceiling)
#   labrag-chexbert14, labrag-devices  the label ladder: the same classifier, only
#                   CheXbert-14 / CheXbert-14 + device types shown and ranked on
#   *.same-reader   (MIMIC only) examples only from the target's radiologist
#   template, template-oracle  no LLM: the shown labels rendered as sentences
# Corpus = train only, as on val.
set -euo pipefail
cd "$(dirname "$0")/.."

s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes "$@"; }
SET=test
MODEL=qwen3.5-27b-vision
SAMPLING="qwen3.5-thinking-32k qwen3.5-thinking-32k-seed2 qwen3.5-thinking-32k-seed3"
SEC=findings-impression
JOINT=findings+impression
POOL=chexpertplus-mimic-rexgradient
MI=$POOL/raddino.ft-attn-seed1/$SEC/chexbert14.devices.incidentals.mc
PNO=joint-no-context-negatives-indication-prior-changes-image
PNC=joint-naive-negatives-indication-prior-changes-image
PLO=joint-labels-only-negatives-indication-prior-changes-image
PSC=joint-simple-negatives-indication-prior-changes-image

# Built by rebuild-labels.sh and earlier runs: all skipped.
labels=""
declare -A lab rule
for dataset in chexpertplus mimic rexgradient; do
  for section in findings impression; do
    lab[$dataset.$section]=$(s --gres=gpu:1 -- labrag labels --dataset $dataset --section $section)
    labels+=":${lab[$dataset.$section]}"
    for labeler in devices incidentals; do
      rule[$dataset.$section.$labeler]=$(s --cpus-per-task=8 -- labrag labels --dataset $dataset --section $section --labeler $labeler)
      labels+=":${rule[$dataset.$section.$labeler]}"
    done
  done
done
ft=$(s -d afterok$labels --gres=gpu:4 --cpus-per-task=144 --mem=256G -- labrag finetune --dataset $POOL --section $SEC --seed 1 --multiclass --labeler chexbert14.devices.incidentals)
ids="labels ${labels#:} finetune $ft"

gens() { for p in $2; do echo -n "$p/$MODEL.$1/$SET "; done; }
# pool <name> <prompt dir>: NAME=run,run,run over the sampling seeds
pool() {
  local r=""
  for sampling in $SAMPLING; do r+=" $(gens $sampling "$2")"; done
  echo -n "$1=$(echo $r | tr ' ' ',') "
}
for dataset in mimic rexgradient; do
  F=$SET/$dataset/$JOINT
  B="labrag build-prompts --experiment $SET --dataset $dataset --section $JOINT"
  pre="$ft"
  for labeler in chexbert14 devices incidentals; do
    after=":${lab[$dataset.findings]}:${lab[$dataset.impression]}"
    if [ $labeler != chexbert14 ]; then
      after=":${rule[$dataset.findings.$labeler]}:${rule[$dataset.impression.$labeler]}"
    fi
    j=$(s -d afterok$after --cpus-per-task=8 -- labrag labels --dataset $dataset --section $JOINT --labeler $labeler)
    pre+=":$j"
  done
  emb=$(s --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag embed --dataset $dataset)
  prd=$(s -d afterok:$ft --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag predict --dataset $dataset --classifier $MI)
  pre+=":$emb:$prd"
  ids+=" $dataset: labels/embed/predict ${pre#*:}"
  MIL=$F/labrag.$POOL.ft-attn-seed1.$SEC.chexbert14.devices.incidentals.mc
  SHL=$F/labrag-shuffled.$POOL.ft-attn-seed1.$SEC.chexbert14.devices.incidentals.mc
  ORL=$F/labrag-oracle.chexbert14.devices.incidentals
  dep=""
  # cell <prompt dir> <build-prompts args>: prompts, then generate + evaluate per
  # sampling seed; PRO = the prompt job
  cell() {
    local dir=$1 g e
    shift
    PRO=$(s -d afterok:$pre --cpus-per-task=8 -- $B "$@")
    ids+=" prompts $PRO"
    for sampling in $SAMPLING; do
      g=$(s -d afterok:$PRO --gres=gpu:1 -- labrag generate --prompts $dir --set $SET --model $MODEL --sampling $sampling)
      e=$(s -d afterok:$g --gres=gpu:1 -- labrag evaluate --generations $(gens $sampling $dir))
      ids+=" $g $e"
      dep+=":$e"
    done
  }
  # template <prompt dir>: the labels the prompts show, rendered; after cell's PRO
  template() {
    local t e
    t=$(s -d afterok:$PRO --cpus-per-task=4 -- labrag template --prompts $1 --set $SET)
    e=$(s -d afterok:$t --gres=gpu:1 -- labrag evaluate --generations $1/template/$SET)
    ids+=" template $t $e"
    dep+=":$e"
  }
  L="--predictions $MI --devices --incidentals"
  cell $F/no-context/$PNO --condition no-context --prompt $PNO
  cell $F/retrieval-only/$PNC --condition retrieval-only --prompt $PNC
  cell $MIL.devices.incidentals/$PNC --condition labrag --prompt $PNC $L
  cell $MIL.devices.incidentals/$PLO --condition labrag --prompt $PLO $L
  cell $MIL.devices.incidentals/$PSC --condition labrag --prompt $PSC $L
  template $MIL.devices.incidentals/$PSC
  cell $SHL.devices.incidentals/$PSC --condition labrag-shuffled --prompt $PSC $L
  cell $ORL/$PSC --condition labrag-oracle --prompt $PSC --devices --incidentals
  template $ORL/$PSC
  cell $MIL/$PSC --condition labrag --prompt $PSC --predictions $MI
  cell $MIL.devices/$PSC --condition labrag --prompt $PSC --predictions $MI --devices
  cells="$(pool no-context $F/no-context/$PNO)"
  cells+="$(pool retrieval-only $F/retrieval-only/$PNC)"
  cells+="$(pool filter-only $MIL.devices.incidentals/$PNC)"
  cells+="$(pool labels-only $MIL.devices.incidentals/$PLO)"
  cells+="$(pool labrag $MIL.devices.incidentals/$PSC)"
  cells+="$(pool labrag-shuffled $SHL.devices.incidentals/$PSC)"
  cells+="$(pool oracle $ORL/$PSC)"
  cells+="$(pool labrag-chexbert14 $MIL/$PSC)"
  cells+="$(pool labrag-devices $MIL.devices/$PSC)"
  cells+="template=$MIL.devices.incidentals/$PSC/template/$SET "
  cells+="template-oracle=$ORL/$PSC/template/$SET "
  if [ $dataset = mimic ]; then
    cell $F/retrieval-only.same-reader/$PNC --condition retrieval-only --prompt $PNC --same reader
    cell $MIL.devices.incidentals.same-reader/$PSC --condition labrag --prompt $PSC $L --same reader
    cells+="$(pool retrieval-only.same-reader $F/retrieval-only.same-reader/$PNC)"
    cells+="$(pool labrag.same-reader $MIL.devices.incidentals.same-reader/$PSC)"
  fi
  cmp=$(s -d afterok$dep -- labrag compare --generations $cells --out $F/qwen3.5-27b.core/$SET)
  ids+=" compare $cmp"
done
echo "$ids"
