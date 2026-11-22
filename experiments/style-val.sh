#!/usr/bin/env bash
# Report-style clusters as a reader stand-in, on MIMIC-CXR val (the only dataset with
# true readers to check against), Findings+Impression, 2 sampling seeds, with the
# pertinent-negatives instruction and the target's indication as in joint-val.
# Retrieval-only with examples from the target's style cluster (styles stage, resid
# k30; the target's own reference assigns it: a style oracle) vs from the target's
# reader (reader-val's) and unrestricted (joint-val's). Qwen3.5-27B with the image.
set -euo pipefail
cd "$(dirname "$0")/.."

s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes "$@"; }
SET=val
MODEL=qwen3.5-27b-vision
SAMPLING="qwen3.5-thinking-32k qwen3.5-thinking-32k-seed2"
JOINT=findings+impression
PN=joint-naive-negatives-indication-image
dataset=mimic
F=val/$dataset/$JOINT
B="labrag build-prompts --experiment val --dataset $dataset --section $JOINT"

lf=$(s --gres=gpu:1 -- labrag labels --dataset $dataset --section findings)
li=$(s --gres=gpu:1 -- labrag labels --dataset $dataset --section impression)
lj=$(s -d afterok:$lf:$li --cpus-per-task=8 -- labrag labels --dataset $dataset --section $JOINT)
sty=$(s -d afterok:$lj --cpus-per-task=8 --mem=64G -- labrag styles --dataset $dataset --section $JOINT)
emb=$(s --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag embed --dataset $dataset)
ids="labels $lf $li $lj styles $sty embed $emb"

gens() { for p in $2; do echo -n "$p/$MODEL.$1/$SET "; done; }
# pool <name> <prompt dir>: NAME=run,run over the sampling seeds
pool() {
  local r=""
  for sampling in $SAMPLING; do r+=" $(gens $sampling "$2")"; done
  echo -n "$1=$(echo $r | tr ' ' ',') "
}
dep=""
cells=""
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
RO=$F/retrieval-only
cell retrieval-only.negatives.indication $lj:$emb $RO/$PN --condition retrieval-only --prompt $PN
cell retrieval-only.negatives.indication.same-reader $lj:$emb $RO.same-reader/$PN --condition retrieval-only --prompt $PN --same reader
cell retrieval-only.negatives.indication.same-style $lj:$emb:$sty $RO.same-style/$PN --condition retrieval-only --prompt $PN --same style
cmp=$(s -d afterok$dep -- labrag compare --generations $cells --out $F/qwen3.5-27b.style/$SET)
ids+=" compare $cmp"
echo "$ids"
