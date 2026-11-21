#!/usr/bin/env bash
# The indication in the prompt on all of MIMIC-CXR and ReXGradient val, Findings, 2
# sampling seeds: negatives-val's three stating-negatives cells (retrieval-only and the
# multiclass pooled fine-tune's positives with the pertinent-negatives instruction, and
# the positives plus the labels predicted absent), each again with the target's
# indication (*-indication-image) and with every example's indication as well
# (*-indication-examples-image). Retrieval, labels and examples are the same as without
# it; only the prompt differs. Compared pooled over both sampling seeds with
# negatives-val's cells, plain retrieval-only, positives only and the oracle.
# Qwen3.5-27B with the image (518 px). Labels, embeds, the fine-tune, predicts and every
# cell negatives-val built are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes "$@"; }
SET=val
MODEL=qwen3.5-27b-vision
SAMPLING="qwen3.5-thinking-32k qwen3.5-thinking-32k-seed2"
SEC=findings-impression
POOL=chexpertplus-mimic-rexgradient
MC=$POOL/raddino.ft-attn-seed1/$SEC/chexbert14.mc
P=section-simple-image

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
  F=val/$dataset/findings
  B="labrag build-prompts --experiment val --dataset $dataset --section findings"
  lf=${lab[$dataset.findings]}
  emb=$(s --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag embed --dataset $dataset)
  prm=$(s -d afterok:$mc --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag predict --dataset $dataset --classifier $MC)
  ids+=" $dataset: embed $emb predict $prm"
  RO=$F/retrieval-only
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
  for p in section-naive-image section-naive-negatives-image \
    section-naive-negatives-indication-image section-naive-negatives-indication-examples-image; do
    cell $lf:$emb $RO/$p --condition retrieval-only --prompt $p
  done
  for p in $P section-simple-negatives-image section-simple-negatives-indication-image \
    section-simple-negatives-indication-examples-image section-simple-absent-image \
    section-simple-absent-indication-image section-simple-absent-indication-examples-image; do
    cell $lf:$prm:$emb $MCL/$p --condition labrag --predictions $MC --prompt $p
  done
  cell $lf:$emb $OR --condition labrag-oracle --prompt $P
  cells="$(pool retrieval-only $RO/section-naive-image)"
  cells+="$(pool retrieval-only.negatives $RO/section-naive-negatives-image)"
  cells+="$(pool retrieval-only.negatives.indication $RO/section-naive-negatives-indication-image)"
  cells+="$(pool retrieval-only.negatives.indication-examples $RO/section-naive-negatives-indication-examples-image)"
  cells+="$(pool mc.positives $MCL/$P)"
  cells+="$(pool mc.positives.negatives $MCL/section-simple-negatives-image)"
  cells+="$(pool mc.positives.negatives.indication $MCL/section-simple-negatives-indication-image)"
  cells+="$(pool mc.positives.negatives.indication-examples $MCL/section-simple-negatives-indication-examples-image)"
  cells+="$(pool mc.positives.absent $MCL/section-simple-absent-image)"
  cells+="$(pool mc.positives.absent.indication $MCL/section-simple-absent-indication-image)"
  cells+="$(pool mc.positives.absent.indication-examples $MCL/section-simple-absent-indication-examples-image)"
  cells+="$(pool oracle $OR)"
  cmp=$(s -d afterok$dep -- labrag compare --generations $cells --out $F/qwen3.5-27b.indication/$SET)
  ids+=" compare $cmp"
done
echo "$ids"
