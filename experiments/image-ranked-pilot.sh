#!/usr/bin/env bash
# Image-ranked control on the pilot: the predicted labels (pooled fine-tune, seed 1)
# and the oracle labels in the prompt, with the examples ranked by image similarity as
# in retrieval-only instead of by label overlap. Next to the label-ranked runs and
# retrieval-only, so ranking and the label list separate: image-ranked − RO = the
# labels alone, label-ranked − image-ranked = the label ranking. Qwen3.5-27B with the
# image (518 px), 4 sampling seeds per condition. Everything already built (labels,
# embed, fine-tune, predict, the label-ranked and RO runs) is skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes "$@"; }
B="labrag build-prompts --experiment pilot --section findings"
F=pilot/chexpertplus/findings
SET=pilot
P=section-simple-image
MODEL=qwen3.5-27b-vision
SAMPLING="qwen3.5-thinking-32k qwen3.5-thinking-32k-seed2 qwen3.5-thinking-32k-seed3 qwen3.5-thinking-32k-seed4"
SEC=findings-impression
POOL=chexpertplus-mimic-rexgradient
FT=$POOL/raddino.ft-attn-seed1/$SEC/chexbert14

labels=""
declare -A lab
for dataset in chexpertplus mimic rexgradient; do
  for section in findings impression; do
    lab[$dataset.$section]=$(s --gres=gpu:1 -- labrag labels --dataset $dataset --section $section)
    labels+=":${lab[$dataset.$section]}"
  done
done
lf=${lab[chexpertplus.findings]}
emb=$(s -w kg35-nvl01 --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag embed --dataset chexpertplus)
ft=$(s -d afterok$labels --gres=gpu:4 --cpus-per-task=144 --mem=256G -- labrag finetune --dataset $POOL --section $SEC --seed 1)
pr=$(s -d afterok:$ft --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag predict --dataset chexpertplus --classifier $FT)
ids="labels ${labels#:} embed $emb finetune $ft predict $pr"

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

FTL=$F/labrag.$POOL.ft-attn-seed1.$SEC.chexbert14
OR=$F/labrag-oracle.chexbert14
RO=$F/retrieval-only/section-naive-image
prompts $pr:$emb $FTL/$P --condition labrag --predictions $FT --prompt $P
prompts $pr:$emb $FTL.image-ranked/$P --condition labrag --predictions $FT --prompt $P --ranking image
prompts $lf:$emb $OR/$P --condition labrag-oracle --prompt $P
prompts $lf:$emb $OR.image-ranked/$P --condition labrag-oracle --prompt $P --ranking image
prompts $lf:$emb $RO --condition retrieval-only --prompt section-naive-image

# pool <name> <prompt dir>: NAME=run,run,... over the sampling seeds
pool() {
  local r=""
  for sampling in $SAMPLING; do r+=" $(gens $sampling "$2")"; done
  echo -n "$1=$(echo $r | tr ' ' ',') "
}
cells=$(pool ft-pooled $FTL/$P)
cells+=$(pool ft-pooled.image-ranked $FTL.image-ranked/$P)
cells+=$(pool oracle $OR/$P)
cells+=$(pool oracle.image-ranked $OR.image-ranked/$P)
cells+=$(pool retrieval-only $RO)
cmp=$(s -d afterok$dep -- labrag compare --generations $cells --out $F/qwen3.5-27b.image-ranked/$SET)
echo "$ids compare $cmp"
