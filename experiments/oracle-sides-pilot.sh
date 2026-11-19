#!/usr/bin/env bash
# Oracle-with-sides ceiling on the pilot: the reference labels, each shown with its side
# from the report's RadGraph-XL graph ("Pleural Effusion (left)") for the target and the
# examples, against the same oracle without sides (identical retrieval) and
# retrieval-only. Tests whether sided labels cut location errors before any sided
# classifier is built. Qwen3.5-27B with the image (518 px), 4 sampling seeds per
# condition. Already built stages are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes "$@"; }
B="labrag build-prompts --experiment pilot --section findings"
F=pilot/chexpertplus/findings
SET=pilot
P=section-simple-image
MODEL=qwen3.5-27b-vision
SAMPLING="qwen3.5-thinking-32k qwen3.5-thinking-32k-seed2 qwen3.5-thinking-32k-seed3 qwen3.5-thinking-32k-seed4"

lf=$(s --gres=gpu:1 -- labrag labels --dataset chexpertplus --section findings)
emb=$(s -w kg35-nvl01 --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag embed --dataset chexpertplus)
sd=$(s -d afterok:$lf --cpus-per-task=8 --mem=32G -- labrag sides --dataset chexpertplus --section findings)
ids="labels $lf embed $emb sides $sd"

gens() { for p in $2; do echo -n "$p/$MODEL.$1/$SET "; done; }
dep=""
# runs <prompts job> <prompt dir>: generate + evaluate per sampling seed
runs() {
  for sampling in $SAMPLING; do
    g=$(s -d afterok:$1 --gres=gpu:1 -- labrag generate --prompts $2 --set $SET --model $MODEL --sampling $sampling)
    e=$(s -d afterok:$g --gres=gpu:1 -- labrag evaluate --generations $(gens $sampling $2))
    ids+=" $g $e"
    dep+=":$e"
  done
}
# prompts <after> <prompt dir> <build-prompts args>: build, then runs
prompts() {
  local after=$1 dir=$2
  shift 2
  local pro
  pro=$(s -d afterok:$after --cpus-per-task=8 -- $B "$@")
  ids+=" prompts $pro"
  runs $pro $dir
}

OR=$F/labrag-oracle.chexbert14
RO=$F/retrieval-only/section-naive-image
prompts $lf:$emb $OR/$P --condition labrag-oracle --prompt $P
prompts $sd:$emb $OR.sides/$P --condition labrag-oracle --prompt $P --sides
prompts $lf:$emb $RO --condition retrieval-only --prompt section-naive-image

# pool <name> <prompt dir>: NAME=run,run,... over the sampling seeds
pool() {
  local r=""
  for sampling in $SAMPLING; do r+=" $(gens $sampling "$2")"; done
  echo -n "$1=$(echo $r | tr ' ' ',') "
}
cells=$(pool oracle $OR/$P)
cells+=$(pool oracle.sides $OR.sides/$P)
cells+=$(pool retrieval-only $RO)
cmp=$(s -d afterok$dep -- labrag compare --generations $cells --out $F/qwen3.5-27b.oracle-sides/$SET)
echo "$ids compare $cmp"
