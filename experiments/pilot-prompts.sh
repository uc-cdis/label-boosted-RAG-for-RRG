#!/usr/bin/env bash
# Prompt tuning: noprior-* (no references to prior studies) for retrieval-only, oracle
# and attn labels; noprior-nocopy-simple (examples for style only) for oracle and attn.
# Each next to its preprint prompt, both sampling seeds. Usage: pilot-prompts.sh <set>
# (pilot or prompt_tuning). Artifacts already built (the preprint-prompt pilot runs) skip.
set -euo pipefail
cd "$(dirname "$0")/.."
SET=${1:?usage: $0 pilot|prompt_tuning}

s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes "$@"; }
B="labrag build-prompts --experiment pilot --section findings"
F=pilot/chexpertplus/findings
MODEL=qwen3.5-27b
CLF=attn-nvfp4-seed1
RO=$F/retrieval-only
OR=$F/labrag-oracle.chexbert14.u0
AT=$F/labrag.$CLF.chexbert14.u0

p1=$(s -- $B --condition retrieval-only --prompt noprior-naive)
p2=$(s -- $B --condition labrag-oracle --prompt noprior-simple)
p3=$(s -- $B --condition labrag-oracle --prompt noprior-nocopy-simple)
p4=$(s -- $B --condition labrag --classifier $CLF --prompt noprior-simple)
p5=$(s -- $B --condition labrag --classifier $CLF --prompt noprior-nocopy-simple)
PROMPTS="$RO/preprint-naive $RO/noprior-naive
  $OR/preprint-simple $OR/noprior-simple $OR/noprior-nocopy-simple
  $AT/preprint-simple $AT/noprior-simple $AT/noprior-nocopy-simple"

ids=""
runs=""
dep=""
for sampling in qwen3.5-thinking-32k qwen3.5-thinking-32k-seed2; do
  g=$(s -d afterok:$p1:$p2:$p3:$p4:$p5 --gres=gpu:1 -- labrag generate --prompts $PROMPTS --set $SET --model $MODEL --sampling $sampling)
  gens=$(for p in $PROMPTS; do echo -n "$p/$MODEL.$sampling/$SET "; done)
  e=$(s -d afterok:$g --gres=gpu:1 -- labrag evaluate --generations $gens)
  ids+=" $g $e"
  dep+=":$e"
  runs+=" $gens"
done
cmp=$(s -d afterok$dep -- labrag compare --generations $runs --out $F/$MODEL.prompts/$SET)
echo "prompts $p1 $p2 $p3 $p4 $p5 generate/evaluate$ids compare $cmp"
