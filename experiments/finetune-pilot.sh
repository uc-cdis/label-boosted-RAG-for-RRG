#!/usr/bin/env bash
# Generation pilot for the fine-tuned classifiers (RAD-DINO + attention head, LP-FT,
# seed 1): trained on CheXpert Plus alone and on CheXpert Plus + MIMIC-CXR +
# ReXGradient, Findings labels predicted with max-F1 thresholds on CheXpert Plus val;
# corpus labels are the reference. Next to retrieval-only and oracle labels.
# Qwen3.5-27B with the image (518 px), 4 sampling seeds per condition, own vLLM server
# per generate and per GREEN judge. Labels, fine-tunes and predicts already built are
# skipped. The CLS embed runs on kg35-nvl01 (full image tree, page cache).
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
FT=raddino.ft-attn-seed1/$SEC/chexbert14

labels=""
for dataset in chexpertplus mimic rexgradient; do
  for section in findings impression; do
    labels+=":$(s --gres=gpu:1 -- labrag labels --dataset $dataset --section $section)"
  done
done
lf=${labels#:}; lf=${lf%%:*}  # chexpertplus findings
emb=$(s -w kg35-nvl01 --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag embed --dataset chexpertplus)
ids="labels ${labels//:/ } embed $emb"

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

# ft_dir <training datasets>: its labrag prompt dir
ft_dir() { echo "$F/labrag.$1.ft-attn-seed1.$SEC.chexbert14/$P"; }
for pool in chexpertplus chexpertplus-mimic-rexgradient; do
  ft=$(s -d afterok$labels --gres=gpu:4 --cpus-per-task=144 --mem=256G -- labrag finetune --dataset $pool --section $SEC --seed 1)
  pr=$(s -d afterok:$ft --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag predict --dataset chexpertplus --classifier $pool/$FT)
  pro=$(s -d afterok:$pr:$emb --cpus-per-task=8 -- $B --condition labrag --predictions $pool/$FT --prompt $P)
  ids+=" finetune $ft predict $pr prompts $pro"
  runs $pro $(ft_dir $pool)
done

RO=$F/retrieval-only/section-naive-image
OR=$F/labrag-oracle.chexbert14/$P
pro=$(s -d afterok:$lf:$emb --cpus-per-task=8 -- $B --condition retrieval-only --prompt section-naive-image)
ids+=" $pro"
runs $pro $RO
pro=$(s -d afterok:$lf:$emb --cpus-per-task=8 -- $B --condition labrag-oracle --prompt $P)
ids+=" $pro"
runs $pro $OR

# pool <name> <prompt dir>: NAME=run,run,... over the sampling seeds
pool() {
  local r=""
  for sampling in $SAMPLING; do r+=" $(gens $sampling "$2")"; done
  echo -n "$1=$(echo $r | tr ' ' ',') "
}
cells=$(pool ft-chexpertplus $(ft_dir chexpertplus))
cells+=$(pool ft-pooled $(ft_dir chexpertplus-mimic-rexgradient))
cells+=$(pool retrieval-only $RO)
cells+=$(pool oracle $OR)
cmp=$(s -d afterok$dep -- labrag compare --generations $cells --out $F/qwen3.5-27b.finetune/$SET)
echo "$ids compare $cmp"
