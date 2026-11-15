#!/usr/bin/env bash
# After pilot-attn-formats.sh, two things at once:
# 1. controls for the attention head's AUROC gain over lr: the same head on the CLS token
#    only (attn-cls, 3 seeds) and lr with standardized features and C tuned on val;
# 2. the pilot generated from the head's labels (nvfp4, seed 1), both sampling seeds,
#    compared with the oracle, U0 (lr labels, both seeds) and retrieval-only.
# The classify jobs read the tokens file, so they run where its page cache is.
set -euo pipefail
cd "$(dirname "$0")/.."

NODE=kg35-nvl01
s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes "$@"; }
C="labrag classify --dataset chexpertplus --section findings"
F=pilot/chexpertplus/findings
MODEL=qwen3.5-27b
RUN=$MODEL.qwen3.5-thinking-32k/pilot
RUN2=$MODEL.qwen3.5-thinking-32k-seed2/pilot
RO=$F/retrieval-only/preprint-naive
U0=$F/labrag.chexbert14.u0/preprint-simple
OR=$F/labrag-oracle.chexbert14.u0/preprint-simple
AT=$F/labrag.attn-nvfp4-seed1.chexbert14.u0/preprint-simple

clf=""
for seed in 1 2 3; do
  clf+=" $(s -w $NODE --gres=gpu:1 --cpus-per-task=16 --mem=64G -- $C --classifier attn-cls --tokens nvfp4 --seed $seed)"
done
clf+=" $(s --cpus-per-task=16 --mem=32G -- $C --classifier lr-tuned)"

pat=$(s -- labrag build-prompts --experiment pilot --section findings --condition labrag --classifier attn-nvfp4-seed1)
g1=$(s -d afterok:$pat --gres=gpu:1 -- labrag generate --prompts $AT --set pilot --model $MODEL --sampling qwen3.5-thinking-32k)
g2=$(s -d afterok:$pat --gres=gpu:1 -- labrag generate --prompts $AT --set pilot --model $MODEL --sampling qwen3.5-thinking-32k-seed2)
ev=$(s -d afterok:$g1:$g2 --gres=gpu:1 -- labrag evaluate --generations $AT/$RUN $AT/$RUN2)
cmp=$(s -d afterok:$ev -- labrag compare --generations $OR/$RUN $AT/$RUN $AT/$RUN2 $U0/$RUN $U0/$RUN2 $RO/$RUN --out $F/$MODEL.attn/pilot)
echo "classify$clf prompts $pat generate $g1 $g2 evaluate $ev compare $cmp"
