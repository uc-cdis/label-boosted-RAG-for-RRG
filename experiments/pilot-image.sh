#!/usr/bin/env bash
# Do labels still help when the model sees the image? Qwen3.5-27B with the frontal
# image at 518 px (shorter side): image only (no-context), retrieval-only, attn labels
# and oracle labels, each next to its text-only prompt (section-*: the same text without
# the image). Control for the 518 choice: image only and attn labels at native
# resolution. Both sampling seeds, on the pilot set. Peak 6 GPUs (3 per seed).
set -euo pipefail
cd "$(dirname "$0")/.."

s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes "$@"; }
B="labrag build-prompts --experiment pilot --section findings"
F=pilot/chexpertplus/findings
CLF=attn-nvfp4-seed1
SET=pilot
RO=$F/retrieval-only
OR=$F/labrag-oracle.chexbert14.u0
AT=$F/labrag.$CLF.chexbert14.u0

p1=$(s -- $B --condition no-context --prompt section-image)
p2=$(s -- $B --condition retrieval-only --prompt section-naive-image)
p3=$(s -- $B --condition labrag-oracle --prompt section-simple-image)
p4=$(s -- $B --condition labrag --classifier $CLF --prompt section-simple-image)
p5=$(s -- $B --condition retrieval-only --prompt section-naive)
p6=$(s -- $B --condition labrag-oracle --prompt section-simple)
p7=$(s -- $B --condition labrag --classifier $CLF --prompt section-simple)
IMAGE="$F/no-context/section-image $RO/section-naive-image
  $OR/section-simple-image $AT/section-simple-image"
NATIVE="$F/no-context/section-image $AT/section-simple-image"
TEXT="$RO/section-naive $OR/section-simple $AT/section-simple"

gens() { for p in $2; do echo -n "$p/$1/$SET "; done; }
ids=""
runs=""
dep=""
for sampling in qwen3.5-thinking-32k qwen3.5-thinking-32k-seed2; do
  gi=$(s -d afterok:$p1:$p2:$p3:$p4 --gres=gpu:1 -- labrag generate --prompts $IMAGE --set $SET --model qwen3.5-27b-vision --sampling $sampling)
  gn=$(s -d afterok:$p1:$p4 --gres=gpu:1 -- labrag generate --prompts $NATIVE --set $SET --model qwen3.5-27b-vision-native --sampling $sampling)
  gt=$(s -d afterok:$p5:$p6:$p7 --gres=gpu:1 -- labrag generate --prompts $TEXT --set $SET --model qwen3.5-27b --sampling $sampling)
  g="$(gens qwen3.5-27b-vision.$sampling "$IMAGE") $(gens qwen3.5-27b-vision-native.$sampling "$NATIVE") $(gens qwen3.5-27b.$sampling "$TEXT")"
  e=$(s -d afterok:$gi:$gn:$gt --gres=gpu:1 -- labrag evaluate --generations $g)
  ids+=" $gi $gn $gt $e"
  dep+=":$e"
  runs+=" $g"
done
cmp=$(s -d afterok$dep -- labrag compare --generations $runs --out $F/qwen3.5-27b.image/$SET)
echo "prompts $p1 $p2 $p3 $p4 $p5 $p6 $p7 generate (518, native, text)/evaluate$ids compare $cmp"
