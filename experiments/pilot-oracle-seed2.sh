#!/usr/bin/env bash
# After pilot.sh: the oracle-label ceiling and a second-seed repeat of U0 (noise control).
set -euo pipefail
cd "$(dirname "$0")/.."

s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes "$@"; }
F=pilot/chexpertplus/findings
MODEL=qwen3.5-27b
RUN=$MODEL.qwen3.5-thinking-32k/pilot
RUN2=$MODEL.qwen3.5-thinking-32k-seed2/pilot
RO=$F/retrieval-only/preprint-naive
U0=$F/labrag.chexbert14.u0/preprint-simple
OR=$F/labrag-oracle.chexbert14.u0/preprint-simple

por=$(s -- labrag build-prompts --experiment pilot --section findings --condition labrag-oracle)
gor=$(s -d afterok:$por --gres=gpu:1 -- labrag generate --prompts $OR --set pilot --model $MODEL --sampling qwen3.5-thinking-32k)
gs2=$(s --gres=gpu:1 -- labrag generate --prompts $U0 --set pilot --model $MODEL --sampling qwen3.5-thinking-32k-seed2)
ev=$(s -d afterok:$gor:$gs2 --gres=gpu:1 -- labrag evaluate --generations $OR/$RUN $U0/$RUN2)
cmp=$(s -d afterok:$ev -- labrag compare --generations $OR/$RUN $U0/$RUN $U0/$RUN2 $RO/$RUN --out $F/$MODEL.oracle-seed2/pilot)
echo "prompts $por generate $gor $gs2 evaluate $ev compare $cmp"
