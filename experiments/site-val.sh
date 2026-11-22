#!/usr/bin/env bash
# Site (the institution that made the study) on ReXGradient val, Findings+Impression,
# 2 sampling seeds, every cell with the pertinent-negatives instruction and the
# target's indication, as in joint-val: the ReXGradient counterpart of reader-val's
# same-reader examples (ReXGradient has no reader). Cells: retrieval-only, mc (the
# pooled multiclass fine-tune's positives) and the oracle (joint-val's, skipped), and
# the same three with examples only from the target's site (--same site). Qwen3.5-27B
# with the image (518 px).
set -euo pipefail
cd "$(dirname "$0")/.."

s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes "$@"; }
SET=val
MODEL=qwen3.5-27b-vision
SAMPLING="qwen3.5-thinking-32k qwen3.5-thinking-32k-seed2"
SEC=findings-impression
JOINT=findings+impression
POOL=chexpertplus-mimic-rexgradient
MC=$POOL/raddino.ft-attn-seed1/$SEC/chexbert14.mc
PN=joint-naive-negatives-indication-image
PS=joint-simple-negatives-indication-image
dataset=rexgradient

labels=""
declare -A lab
for d in chexpertplus mimic rexgradient; do
  for section in findings impression; do
    lab[$d.$section]=$(s --gres=gpu:1 -- labrag labels --dataset $d --section $section)
    labels+=":${lab[$d.$section]}"
  done
done
mc=$(s -d afterok$labels --gres=gpu:4 --cpus-per-task=144 --mem=256G -- labrag finetune --dataset $POOL --section $SEC --seed 1 --multiclass)
ids="labels ${labels#:} finetune $mc"

F=val/$dataset/$JOINT
B="labrag build-prompts --experiment val --dataset $dataset --section $JOINT"
lj=$(s -d afterok:${lab[$dataset.findings]}:${lab[$dataset.impression]} --cpus-per-task=8 -- labrag labels --dataset $dataset --section $JOINT)
emb=$(s --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag embed --dataset $dataset)
prm=$(s -d afterok:$mc --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag predict --dataset $dataset --classifier $MC)
ids+=" labels $lj embed $emb predict $prm"

gens() { for p in $2; do echo -n "$p/$MODEL.$1/$SET "; done; }
# pool <name> <prompt dir>: NAME=run,run over the sampling seeds
pool() {
  local r=""
  for sampling in $SAMPLING; do r+=" $(gens $sampling "$2")"; done
  echo -n "$1=$(echo $r | tr ' ' ',') "
}
dep=""
cells=""
# cell <name> <after> <prompt dir> <build-prompts args>: prompts, then generate +
# evaluate per sampling seed
cell() {
  local name=$1 after=$2 dir=$3
  shift 3
  local pro g e
  pro=$(s -d afterok:$after --cpus-per-task=8 -- $B "$@")
  ids+=" prompts $pro"
  for sampling in $SAMPLING; do
    g=$(s -d afterok:$pro --gres=gpu:1 -- labrag generate --prompts $dir --set $SET --model $MODEL --sampling $sampling)
    e=$(s -d afterok:$g --gres=gpu:1 -- labrag evaluate --generations $(gens $sampling $dir))
    ids+=" $g $e"
    dep+=":$e"
  done
  cells+="$(pool $name $dir)"
}
RO=$F/retrieval-only
MCL=$F/labrag.$POOL.ft-attn-seed1.$SEC.chexbert14.mc
ORD=$F/labrag-oracle.chexbert14
cell retrieval-only.negatives.indication $lj:$emb $RO/$PN --condition retrieval-only --prompt $PN
cell retrieval-only.negatives.indication.same-site $lj:$emb $RO.same-site/$PN --condition retrieval-only --prompt $PN --same site
cell oracle.negatives.indication $lj:$emb $ORD/$PS --condition labrag-oracle --prompt $PS
cell oracle.negatives.indication.same-site $lj:$emb $ORD.same-site/$PS --condition labrag-oracle --prompt $PS --same site
cell mc.positives.negatives.indication $lj:$prm:$emb $MCL/$PS --condition labrag --predictions $MC --prompt $PS
cell mc.positives.negatives.indication.same-site $lj:$prm:$emb $MCL.same-site/$PS --condition labrag --predictions $MC --prompt $PS --same site
cmp=$(s -d afterok$dep -- labrag compare --generations $cells --out $F/qwen3.5-27b.site/$SET)
ids+=" compare $cmp"
echo "$ids"
