#!/usr/bin/env bash
# The classifier on all of a study's images (attn-all-images: every frontal and
# lateral, at most 4, the head attending over all their tokens) vs the preferred
# frontal alone, on MIMIC-CXR and ReXGradient val, Findings+Impression, 2 sampling
# seeds, every cell with the pertinent-negatives instruction and the target's
# indication, as in joint-val. Same studies and retrieval; only the labels change.
# Cells: retrieval-only, mc and the oracle (joint-val's), mc from plain seed 2 (the
# fine-tune noise floor; MIMIC's is reader-val's), mc-all-images; on MIMIC also the
# same-reader cells (reader-val's) and mc-all-images with same-reader examples.
# Qwen3.5-27B with the image (518 px).
set -euo pipefail
cd "$(dirname "$0")/.."

s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes "$@"; }
SET=val
MODEL=qwen3.5-27b-vision
SAMPLING="qwen3.5-thinking-32k qwen3.5-thinking-32k-seed2"
SEC=findings-impression
JOINT=findings+impression
POOL=chexpertplus-mimic-rexgradient
PN=joint-naive-negatives-indication-image
PS=joint-simple-negatives-indication-image
FTS="ft-attn-seed1 ft-attn-seed2 ft-attn-all-images-seed1"
declare -A NAME=([ft-attn-seed1]=mc [ft-attn-seed2]=mc-seed2 [ft-attn-all-images-seed1]=mc-all-images)

labels=""
declare -A lab
for d in chexpertplus mimic rexgradient; do
  for section in findings impression; do
    lab[$d.$section]=$(s --gres=gpu:1 -- labrag labels --dataset $d --section $section)
    labels+=":${lab[$d.$section]}"
  done
done
FT="--gres=gpu:4 --cpus-per-task=144 --mem=256G -- labrag finetune --dataset $POOL --section $SEC --multiclass"
declare -A ft
ft[ft-attn-seed1]=$(s -d afterok$labels $FT --seed 1)
ft[ft-attn-seed2]=$(s -d afterok$labels $FT --seed 2)
ft[ft-attn-all-images-seed1]=$(s -d afterok$labels $FT --seed 1 --classifier attn-all-images)
ids="labels ${labels#:} finetune ${ft[*]}"

gens() { for p in $2; do echo -n "$p/$MODEL.$1/$SET "; done; }
# pool <name> <prompt dir>: NAME=run,run over the sampling seeds
pool() {
  local r=""
  for sampling in $SAMPLING; do r+=" $(gens $sampling "$2")"; done
  echo -n "$1=$(echo $r | tr ' ' ',') "
}
# cell <name> <after> <prompt dir> <build-prompts args>: prompts, then generate +
# evaluate per sampling seed
cell() {
  local name=$1 after=$2 dir=$3
  shift 3
  local pro g e
  pro=$(s -d afterok:$after --cpus-per-task=8 -- $B "$@")
  ids+=" prompts $pro"
  for sampling in $SAMPLING; do
    g=$(s -d afterok:$pro --gres=gpu:1 -- labrag generate --prompts $dir --set $SET --model $MODEL --sampling $sampling)
    e=$(s -d afterok:$g --gres=gpu:1 -- labrag evaluate --generations $(gens $sampling $dir))
    ids+=" $g $e"
    dep+=":$e"
  done
  cells+="$(pool $name $dir)"
}
mc() { echo "--condition labrag --predictions $POOL/raddino.$1/$SEC/chexbert14.mc --prompt $PS"; }

for dataset in mimic rexgradient; do
  F=val/$dataset/$JOINT
  B="labrag build-prompts --experiment val --dataset $dataset --section $JOINT"
  lj=$(s -d afterok:${lab[$dataset.findings]}:${lab[$dataset.impression]} --cpus-per-task=8 -- labrag labels --dataset $dataset --section $JOINT)
  emb=$(s --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag embed --dataset $dataset)
  declare -A prm
  for name in $FTS; do
    prm[$name]=$(s -d afterok:${ft[$name]} --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag predict --dataset $dataset --classifier $POOL/raddino.$name/$SEC/chexbert14.mc)
  done
  ids+=" $dataset: labels $lj embed $emb predict ${prm[*]}"
  L=$F/labrag.$POOL
  dep=""
  cells=""
  cell retrieval-only.negatives.indication $lj:$emb $F/retrieval-only/$PN --condition retrieval-only --prompt $PN
  cell oracle.negatives.indication $lj:$emb $F/labrag-oracle.chexbert14/$PS --condition labrag-oracle --prompt $PS
  for name in $FTS; do
    cell ${NAME[$name]}.positives.negatives.indication $lj:$emb:${prm[$name]} $L.$name.$SEC.chexbert14.mc/$PS $(mc $name)
  done
  if [[ $dataset == mimic ]]; then
    cell retrieval-only.negatives.indication.same-reader $lj:$emb $F/retrieval-only.same-reader/$PN --condition retrieval-only --prompt $PN --same reader
    cell oracle.negatives.indication.same-reader $lj:$emb $F/labrag-oracle.chexbert14.same-reader/$PS --condition labrag-oracle --prompt $PS --same reader
    for name in ft-attn-seed1 ft-attn-all-images-seed1; do
      cell ${NAME[$name]}.positives.negatives.indication.same-reader $lj:$emb:${prm[$name]} $L.$name.$SEC.chexbert14.mc.same-reader/$PS $(mc $name) --same reader
    done
  fi
  cmp=$(s -d afterok$dep -- labrag compare --generations $cells --out $F/qwen3.5-27b.images/$SET)
  ids+=" compare $cmp"
done
echo "$ids"
