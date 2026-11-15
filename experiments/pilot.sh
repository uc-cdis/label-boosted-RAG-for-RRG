#!/usr/bin/env bash

s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes "$@"; }
F=pilot/chexpertplus/findings
RUN=qwen3.5-27b.qwen3.5-thinking-32k/pilot
RO=$F/retrieval-only/preprint-naive
U0=$F/labrag.chexbert14.u0/preprint-simple

lab=$(s --gres=gpu:1 -- labrag labels --dataset chexpertplus --section findings)
emb=$(s --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag embed --dataset chexpertplus)
c0=$(s -d afterok:$lab:$emb --cpus-per-task=16 -- labrag classify --dataset chexpertplus --section findings)
pro=$(s -d afterok:$lab:$emb -- labrag build-prompts --experiment pilot --section findings --condition retrieval-only)
p0=$(s -d afterok:$c0 -- labrag build-prompts --experiment pilot --section findings --condition labrag)
gen=$(s -d afterok:$pro:$p0 --gres=gpu:1 -- labrag generate --prompts $RO $U0 --set pilot --model qwen3.5-27b --sampling qwen3.5-thinking-32k)
ev=$(s -d afterok:$gen --gres=gpu:1 -- labrag evaluate --generations $RO/$RUN $U0/$RUN)
cmp=$(s -d afterok:$ev -- labrag compare --generations $U0/$RUN $RO/$RUN --out $F/$RUN)
echo "labels $lab embed $emb classify $c0 prompts $pro $p0 generate $gen evaluate $ev compare $cmp"
