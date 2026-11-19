#!/usr/bin/env bash
# Smoke run on MIMIC-CXR and ReXGradient (build-prompts experiment "smoke": 100 val
# patients each, one study per patient, corpus = train), Findings, the config the
# CheXpert Plus pilot settled on: Qwen3.5-27B with the image (518 px), 1 sampling seed,
# retrieval-only, labrag with the pooled fine-tune's labels (max-F1 on the dataset's own
# val) and the oracle. One generate server and one evaluate per dataset. Checks the
# pipeline, report style, thinking length / truncation and throughput; too small for
# significance. Labels, the fine-tune and its predicts already built are skipped.
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

labels=""
declare -A lab
for dataset in chexpertplus mimic rexgradient; do
  for section in findings impression; do
    lab[$dataset.$section]=$(s --gres=gpu:1 -- labrag labels --dataset $dataset --section $section)
    labels+=":${lab[$dataset.$section]}"
  done
done
ft=$(s -d afterok$labels --gres=gpu:4 --cpus-per-task=144 --mem=256G -- labrag finetune --dataset $POOL --section $SEC --seed 1)
ids="labels ${labels#:} finetune $ft"

for dataset in mimic rexgradient; do
  F=smoke/$dataset/findings
  B="labrag build-prompts --experiment smoke --dataset $dataset --section findings"
  lf=${lab[$dataset.findings]}
  emb=$(s --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag embed --dataset $dataset)
  pr=$(s -d afterok:$ft --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag predict --dataset $dataset --classifier $FT)
  RO=$F/retrieval-only/section-naive-image
  FTL=$F/labrag.$POOL.ft-attn-seed1.$SEC.chexbert14/$P
  OR=$F/labrag-oracle.chexbert14/$P
  p1=$(s -d afterok:$lf:$emb --cpus-per-task=8 -- $B --condition retrieval-only --prompt section-naive-image)
  p2=$(s -d afterok:$lf:$pr:$emb --cpus-per-task=8 -- $B --condition labrag --predictions $FT --prompt $P)
  p3=$(s -d afterok:$lf:$emb --cpus-per-task=8 -- $B --condition labrag-oracle --prompt $P)
  g=$(s -d afterok:$p1:$p2:$p3 --gres=gpu:1 -- labrag generate --prompts $RO $FTL $OR --set $SET --model $MODEL --sampling $SAMPLING)
  run() { echo -n "$1/$MODEL.$SAMPLING/$SET"; }
  e=$(s -d afterok:$g --gres=gpu:1 -- labrag evaluate --generations $(run $RO) $(run $FTL) $(run $OR))
  cmp=$(s -d afterok:$e -- labrag compare --generations retrieval-only=$(run $RO) ft-pooled=$(run $FTL) oracle=$(run $OR) --out $F/qwen3.5-27b.smoke/$SET)
  ids+=" $dataset: embed $emb predict $pr prompts $p1 $p2 $p3 generate $g evaluate $e compare $cmp"
done
echo "$ids"
