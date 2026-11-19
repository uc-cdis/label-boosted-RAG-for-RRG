#!/usr/bin/env bash
# Generation pilot for the fine-tuned classifiers (RAD-DINO + attention head, LP-FT,
# seed 1): trained on CheXpert Plus alone and on CheXpert Plus + MIMIC-CXR +
# ReXGradient, Findings labels predicted with max-F1 thresholds on CheXpert Plus val;
# corpus labels are the reference. Next to the frozen control (the same attention
# head with Findings + Impression readouts on frozen RAD-DINO tokens, CheXpert Plus,
# seed 1), retrieval-only and oracle labels.
# Qwen3.5-27B with the image (518 px), 4 sampling seeds per condition, own vLLM server
# per generate and per GREEN judge. Labels, fine-tunes and predicts already built are
# skipped. Embeds and the frozen head run on kg35-nvl01 (full image tree; the head
# reads the token store through its page cache).
set -euo pipefail
cd "$(dirname "$0")/.."

s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes "$@"; }
B="labrag build-prompts --experiment pilot --dataset chexpertplus --section findings"
F=pilot/chexpertplus/findings
SET=pilot
P=section-simple-image
MODEL=qwen3.5-27b-vision
SAMPLING="qwen3.5-thinking-32k qwen3.5-thinking-32k-seed2 qwen3.5-thinking-32k-seed3 qwen3.5-thinking-32k-seed4"
SEC=findings-impression
FT=raddino.ft-attn-seed1/$SEC/chexbert14

NODE=kg35-nvl01
labels=""
declare -A lab
for dataset in chexpertplus mimic rexgradient; do
  for section in findings impression; do
    lab[$dataset.$section]=$(s --gres=gpu:1 -- labrag labels --dataset $dataset --section $section)
    labels+=":${lab[$dataset.$section]}"
  done
done
lf=${lab[chexpertplus.findings]}
li=${lab[chexpertplus.impression]}
emb=$(s -w $NODE --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag embed --dataset chexpertplus)
tok=$(s -w $NODE --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag embed --dataset chexpertplus --feature tokens)
ids="labels ${labels#:} embed $emb $tok"

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

FROZEN=raddino.attn-nvfp4-seed1
c=$(s -w $NODE -d afterok:$tok:$lf:$li --gres=gpu:1 --cpus-per-task=16 --mem=64G -- labrag classify --dataset chexpertplus --section $SEC --classifier attn --tokens nvfp4 --seed 1)
pro=$(s -d afterok:$c:$emb --cpus-per-task=8 -- $B --condition labrag --classifier chexpertplus/$FROZEN/$SEC/chexbert14 --prompt $P)
ids+=" classify $c prompts $pro"
FR=$F/labrag.${FROZEN#raddino.}.$SEC.chexbert14/$P
runs $pro $FR

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
cells+=$(pool frozen $FR)
cells+=$(pool retrieval-only $RO)
cells+=$(pool oracle $OR)
cmp=$(s -d afterok$dep -- labrag compare --generations $cells --out $F/qwen3.5-27b.finetune/$SET)
echo "$ids compare $cmp"
