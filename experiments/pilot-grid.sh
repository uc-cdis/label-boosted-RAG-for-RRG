#!/usr/bin/env bash
# Classifier grid on the pilot, CheXbert-14 U0 labels: training labels (Findings only vs
# Findings + Impression sections head) × thresholds (max F1, 0.5, prevalence) × 3
# classifier seeds × 2 sampling seeds, Qwen3.5-27B with the image (518 px). 0.5 and
# prevalence re-threshold the heads' saved probabilities (CPU). Seed-1 max F1 runs are
# already built. Every run gets its own vLLM server for generate and for the GREEN
# judge (one job each, all in parallel). One pooled compare: each cell = 6 runs, retrieval-only and image only = 2 runs each.
set -euo pipefail
cd "$(dirname "$0")/.."

s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes "$@"; }
B="labrag build-prompts --experiment pilot --section findings"
F=pilot/chexpertplus/findings
SET=pilot
P=section-simple-image
SAMPLING="qwen3.5-thinking-32k qwen3.5-thinking-32k-seed2"
declare -A SUFFIX=([max-f1]="" [0.5]=-t50 [prevalence]=-prev)

gens() { for p in $2; do echo -n "$p/$1/$SET "; done; }
# clf <head: findings|sections> <seed> <rule>: build-prompts --classifier name
clf() {
  local name=attn-nvfp4-seed$2${SUFFIX[$3]}
  [ "$1" = sections ] && name+=-sections
  echo $name
}
# src <head> <seed>: the classify artifact under classifiers/
src() {
  if [ "$1" = sections ]; then
    echo chexpertplus/raddino.attn-nvfp4-seed$2-sections/chexbert14.u0
  else
    echo chexpertplus/raddino.attn-nvfp4-seed$2/findings/chexbert14.u0
  fi
}

ids=""
dep=""
for head in findings sections; do
  for seed in 1 2 3; do
    for rule in max-f1 0.5 prevalence; do
      [ $seed = 1 ] && [ $rule = max-f1 ] && continue  # built
      after=""
      if [ $rule != max-f1 ]; then
        t=$(s --cpus-per-task=8 -- labrag threshold --classifier $(src $head $seed) --rule $rule)
        after="-d afterok:$t"
        ids+=" $t"
      fi
      c=$(clf $head $seed $rule)
      pro=$(s $after --cpus-per-task=8 -- $B --condition labrag --classifier $c --prompt $P)
      ids+=" $pro"
      new=$F/labrag.$c.chexbert14.u0/$P
      for sampling in $SAMPLING; do
        run=qwen3.5-27b-vision.$sampling
        g=$(s -d afterok:$pro --gres=gpu:1 -- labrag generate --prompts $new --set $SET --model qwen3.5-27b-vision --sampling $sampling)
        e=$(s -d afterok:$g --gres=gpu:1 -- labrag evaluate --generations $(gens $run "$new"))
        ids+=" $g $e"
        dep+=":$e"
      done
    done
  done
done

# pool <name> <prompt dirs>: NAME=run,run,... over both sampling seeds
pool() {
  local runs=""
  for sampling in $SAMPLING; do runs+=" $(gens qwen3.5-27b-vision.$sampling "$2")"; done
  echo -n "$1=$(echo $runs | tr ' ' ',') "
}
cells=""
for head in findings sections; do
  for rule in max-f1 0.5 prevalence; do
    dirs=""
    for seed in 1 2 3; do dirs+=" $F/labrag.$(clf $head $seed $rule).chexbert14.u0/$P"; done
    cells+=$(pool $head.$rule "$dirs")
  done
done
cells+=$(pool retrieval-only "$F/retrieval-only/section-naive-image")
cells+=$(pool image-only "$F/no-context/section-image")
cmp=$(s -d afterok$dep -- labrag compare --generations $cells --out $F/qwen3.5-27b.classifier-grid/$SET)
echo "threshold/prompts/generate/evaluate$ids compare $cmp"
