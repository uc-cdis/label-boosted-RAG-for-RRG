#!/usr/bin/env bash
# Classifier grid on the pilot, from an empty shared root. CheXbert-14 labels (U0):
# training labels (Findings only vs one head with Findings + Impression readouts) ×
# thresholds (max F1, 0.5, prevalence) × 3 classifier seeds × 2 sampling seeds,
# Qwen3.5-27B with the image (518 px), next to retrieval-only, image only and oracle
# labels (2 runs each). Both heads read the frontal token store. Embeds and heads run on
# kg35-nvl01: the full image tree is there, and the heads share its page cache. 0.5 and
# prevalence re-threshold the heads' saved probabilities (CPU). Every run gets its own
# vLLM server for generate and for the GREEN judge. One pooled compare: each cell =
# 6 runs.
set -euo pipefail
cd "$(dirname "$0")/.."

NODE=kg35-nvl01
s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes "$@"; }
B="labrag build-prompts --experiment pilot --dataset chexpertplus --section findings"
F=pilot/chexpertplus/findings
SET=pilot
P=section-simple-image
MODEL=qwen3.5-27b-vision
SAMPLING="qwen3.5-thinking-32k qwen3.5-thinking-32k-seed2"
declare -A SUFFIX=([max-f1]="" [0.5]=-t50 [prevalence]=-prev)
declare -A SECTIONS=([findings]=findings [sections]=findings-impression)

lf=$(s --gres=gpu:1 -- labrag labels --dataset chexpertplus --section findings)
li=$(s --gres=gpu:1 -- labrag labels --dataset chexpertplus --section impression)
emb=$(s -w $NODE --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag embed --dataset chexpertplus)
tok=$(s -w $NODE --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag embed --dataset chexpertplus --feature tokens)
ids="labels $lf $li embed $emb $tok"

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

for head in findings sections; do
  sec=${SECTIONS[$head]}
  after=afterok:$tok:$lf
  [ $head = sections ] && after+=:$li
  for seed in 1 2 3; do
    name=raddino.attn-nvfp4-seed$seed
    c=$(s -w $NODE -d $after --gres=gpu:1 --cpus-per-task=16 --mem=64G -- labrag classify --dataset chexpertplus --section $sec --classifier attn --tokens nvfp4 --seed $seed)
    ids+=" classify $c"
    for rule in max-f1 0.5 prevalence; do
      ready=$c
      if [ $rule != max-f1 ]; then
        ready=$(s -d afterok:$c --cpus-per-task=8 -- labrag threshold --classifier chexpertplus/$name/$sec/chexbert14 --rule $rule)
        ids+=" $ready"
      fi
      clf=$name${SUFFIX[$rule]}
      pro=$(s -d afterok:$ready:$emb --cpus-per-task=8 -- $B --condition labrag --classifier chexpertplus/$clf/$sec/chexbert14 --prompt $P)
      ids+=" $pro"
      runs $pro $F/labrag.${clf#raddino.}.$sec.chexbert14/$P
    done
  done
done

RO=$F/retrieval-only/section-naive-image
IO=$F/no-context/section-image
OR=$F/labrag-oracle.chexbert14/$P
pro=$(s -d afterok:$lf:$emb --cpus-per-task=8 -- $B --condition retrieval-only --prompt section-naive-image)
ids+=" $pro"
runs $pro $RO
pro=$(s -d afterok:$lf --cpus-per-task=8 -- $B --condition no-context --prompt section-image)
ids+=" $pro"
runs $pro $IO
pro=$(s -d afterok:$lf:$emb --cpus-per-task=8 -- $B --condition labrag-oracle --prompt $P)
ids+=" $pro"
runs $pro $OR

# pool <name> <prompt dirs>: NAME=run,run,... over both sampling seeds
pool() {
  local r=""
  for sampling in $SAMPLING; do r+=" $(gens $sampling "$2")"; done
  echo -n "$1=$(echo $r | tr ' ' ',') "
}
cells=""
for head in findings sections; do
  sec=${SECTIONS[$head]}
  for rule in max-f1 0.5 prevalence; do
    dirs=""
    for seed in 1 2 3; do
      dirs+=" $F/labrag.attn-nvfp4-seed$seed${SUFFIX[$rule]}.$sec.chexbert14/$P"
    done
    cells+=$(pool $head.$rule "$dirs")
  done
done
cells+=$(pool retrieval-only $RO)
cells+=$(pool image-only $IO)
cells+=$(pool oracle $OR)
cmp=$(s -d afterok$dep -- labrag compare --generations $cells --out $F/qwen3.5-27b.classifier-grid/$SET)
echo "$ids compare $cmp"
