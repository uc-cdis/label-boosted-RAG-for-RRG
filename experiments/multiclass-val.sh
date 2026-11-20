#!/usr/bin/env bash
# Multiclass labels on all of MIMIC-CXR and ReXGradient val (build-prompts experiment
# "val", corpus = train), Findings: the pooled fine-tune (seed 1) trained on CheXbert's
# four values per label (positive, negative, uncertain, blank) and read by argmax,
# shown by category (section-verbose-image), next to its positives alone (plain prompt:
# separates the argmax positives from the categories), the binary pooled fine-tune
# (max-F1 on the dataset's own val, in-sample here), retrieval-only and the oracle.
# Qwen3.5-27B with the image (518 px), 1 sampling seed, one generate and one evaluate
# per condition. Labels, embeds, the binary fine-tune and its predicts already built are
# skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes "$@"; }
SET=val
P=section-simple-image
MODEL=qwen3.5-27b-vision
SAMPLING=qwen3.5-thinking-32k
SEC=findings-impression
POOL=chexpertplus-mimic-rexgradient
FT=$POOL/raddino.ft-attn-seed1/$SEC/chexbert14
MC=$FT.mc

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

run() { echo -n "$1/$MODEL.$SAMPLING/$SET"; }
for dataset in mimic rexgradient; do
  F=val/$dataset/findings
  B="labrag build-prompts --experiment val --dataset $dataset --section findings"
  lf=${lab[$dataset.findings]}
  emb=$(s --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag embed --dataset $dataset)
  PR="--gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag predict --dataset $dataset"
  pr=$(s -d afterok:$ft $PR --classifier $FT)
  prm=$(s -d afterok:$mc $PR --classifier $MC)
  ids+=" $dataset: embed $emb predict $pr $prm"
  RO=$F/retrieval-only/section-naive-image
  FTL=$F/labrag.$POOL.ft-attn-seed1.$SEC.chexbert14/$P
  MCL=$F/labrag.$POOL.ft-attn-seed1.$SEC.chexbert14.mc
  OR=$F/labrag-oracle.chexbert14/$P
  dep=""
  # cell <after> <prompt dir> <build-prompts args>: prompts, generate, evaluate
  cell() {
    local after=$1 dir=$2
    shift 2
    local pro g e
    pro=$(s -d afterok:$after --cpus-per-task=8 -- $B "$@")
    g=$(s -d afterok:$pro --gres=gpu:1 -- labrag generate --prompts $dir --set $SET --model $MODEL --sampling $SAMPLING)
    e=$(s -d afterok:$g --gres=gpu:1 -- labrag evaluate --generations $(run $dir))
    ids+=" prompts $pro generate $g evaluate $e"
    dep+=":$e"
  }
  cell $lf:$emb $RO --condition retrieval-only --prompt section-naive-image
  cell $lf:$pr:$emb $FTL --condition labrag --predictions $FT --prompt $P
  cell $lf:$prm:$emb $MCL/section-verbose-image --condition labrag --predictions $MC --prompt section-verbose-image
  cell $lf:$prm:$emb $MCL/$P --condition labrag --predictions $MC --prompt $P
  cell $lf:$emb $OR --condition labrag-oracle --prompt $P
  cells="retrieval-only=$(run $RO) ft-pooled=$(run $FTL)"
  cells+=" ft-pooled.mc=$(run $MCL/section-verbose-image) ft-pooled.mc.positives=$(run $MCL/$P)"
  cells+=" oracle=$(run $OR)"
  cmp=$(s -d afterok$dep -- labrag compare --generations $cells --out $F/qwen3.5-27b.multiclass/$SET)
  ids+=" compare $cmp"
done
echo "$ids"
