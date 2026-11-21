#!/usr/bin/env bash
# Findings+Impression generated and scored as one text on all of MIMIC-CXR and
# ReXGradient val (studies with both sections), 2 sampling seeds: retrieval-only, the
# multiclass pooled fine-tune's positives (its findings and impression readouts
# combined) and the oracle (the reference findings and impression labels, combined the
# same way), each with the pertinent-negatives instruction and the target's indication
# (joint-*-indication-image), the oracle included this time, so its gap is like-for-like.
# Qwen3.5-27B with the image (518 px). Labels, embeds, the fine-tune and predicts that
# earlier experiments built are skipped; new are the joint labels and the 3 cells.
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
  emb=$(s --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag embed --dataset $dataset)
  prm=$(s -d afterok:$mc --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag predict --dataset $dataset --classifier $MC)
  ids+=" $dataset: labels $lj embed $emb predict $prm"
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
  cell $lj:$emb $RO/$PN --condition retrieval-only --prompt $PN
  cell $lj:$prm:$emb $MCL/$PS --condition labrag --predictions $MC --prompt $PS
  cell $lj:$emb $ORD/$PS --condition labrag-oracle --prompt $PS
  cells="$(pool retrieval-only.negatives.indication $RO/$PN)"
  cells+="$(pool mc.positives.negatives.indication $MCL/$PS)"
  cells+="$(pool oracle.negatives.indication $ORD/$PS)"
  cmp=$(s -d afterok$dep -- labrag compare --generations $cells --out $F/qwen3.5-27b.joint/$SET)
  ids+=" compare $cmp"
done
echo "$ids"
