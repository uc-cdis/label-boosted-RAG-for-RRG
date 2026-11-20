#!/usr/bin/env bash
# Stating negatives on all of MIMIC-CXR and ReXGradient val, Findings, 2 sampling seeds
# (also multiclass-val's second seed): retrieval-only and the multiclass pooled
# fine-tune's positives (argmax), each as before and with one instruction to state
# pertinent negatives (*-negatives-image), and the positives plus the labels predicted
# negative, each to be stated as absent (section-simple-absent-image); next to the
# binary pooled fine-tune and the oracle. Retrieval and positives are the same in every
# multiclass cell; only the prompt differs. Compared pooled over both sampling seeds
# (each run's own mean is a run row).
# Qwen3.5-27B with the image (518 px). Labels, embeds, fine-tunes, predicts, prompts
# and the seed-1 runs multiclass-val built are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes "$@"; }
SET=val
MODEL=qwen3.5-27b-vision
SAMPLING="qwen3.5-thinking-32k qwen3.5-thinking-32k-seed2"
SEC=findings-impression
POOL=chexpertplus-mimic-rexgradient
FT=$POOL/raddino.ft-attn-seed1/$SEC/chexbert14
MC=$FT.mc
P=section-simple-image

labels=""
declare -A lab
for dataset in chexpertplus mimic rexgradient; do
  for section in findings impression; do
    lab[$dataset.$section]=$(s --gres=gpu:1 -- labrag labels --dataset $dataset --section $section)
    labels+=":${lab[$dataset.$section]}"
  done
done
FIT="--gres=gpu:4 --cpus-per-task=144 --mem=256G -- labrag finetune --dataset $POOL --section $SEC --seed 1"
ft=$(s -d afterok$labels $FIT)
mc=$(s -d afterok$labels $FIT --multiclass)
ids="labels ${labels#:} finetune $ft $mc"

gens() { for p in $2; do echo -n "$p/$MODEL.$1/$SET "; done; }
# pool <name> <prompt dir>: NAME=run,run over the sampling seeds
pool() {
  local r=""
  for sampling in $SAMPLING; do r+=" $(gens $sampling "$2")"; done
  echo -n "$1=$(echo $r | tr ' ' ',') "
}
for dataset in mimic rexgradient; do
  F=val/$dataset/findings
  B="labrag build-prompts --experiment val --dataset $dataset --section findings"
  lf=${lab[$dataset.findings]}
  emb=$(s --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag embed --dataset $dataset)
  PR="--gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag predict --dataset $dataset"
  pr=$(s -d afterok:$ft $PR --classifier $FT)
  prm=$(s -d afterok:$mc $PR --classifier $MC)
  ids+=" $dataset: embed $emb predict $pr $prm"
  RO=$F/retrieval-only
  FTL=$F/labrag.$POOL.ft-attn-seed1.$SEC.chexbert14/$P
  MCL=$F/labrag.$POOL.ft-attn-seed1.$SEC.chexbert14.mc
  OR=$F/labrag-oracle.chexbert14/$P
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
  for p in section-naive-image section-naive-negatives-image; do
    cell $lf:$emb $RO/$p --condition retrieval-only --prompt $p
  done
  for p in $P section-simple-negatives-image section-simple-absent-image; do
    cell $lf:$prm:$emb $MCL/$p --condition labrag --predictions $MC --prompt $p
  done
  cell $lf:$pr:$emb $FTL --condition labrag --predictions $FT --prompt $P
  cell $lf:$emb $OR --condition labrag-oracle --prompt $P
  cells="$(pool retrieval-only $RO/section-naive-image)"
  cells+="$(pool retrieval-only.negatives $RO/section-naive-negatives-image)"
  cells+="$(pool mc.positives $MCL/$P)"
  cells+="$(pool mc.positives.negatives $MCL/section-simple-negatives-image)"
  cells+="$(pool mc.positives.absent $MCL/section-simple-absent-image)"
  cells+="$(pool ft-pooled $FTL)$(pool oracle $OR)"
  cmp=$(s -d afterok$dep -- labrag compare --generations $cells --out $F/qwen3.5-27b.negatives/$SET)
  ids+=" compare $cmp"
done
echo "$ids"
