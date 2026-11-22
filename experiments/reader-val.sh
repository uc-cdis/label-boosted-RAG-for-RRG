#!/usr/bin/env bash
# Reader (the radiologist on the report) on MIMIC-CXR val, Findings+Impression, 2
# sampling seeds, every cell with the pertinent-negatives instruction and the target's
# indication, as in joint-val. Two routes for the reader, crossed:
#   labels:   the pooled multiclass fine-tune conditioned on the reader (attn-reader)
#             vs the plain one (seed 1, joint-val's) and a second plain seed (the
#             fine-tune noise floor for that comparison);
#   examples: retrieval restricted to the target's reader (--same-reader) vs all.
# Cells: retrieval-only, mc and the oracle (joint-val's, skipped), retrieval-only, mc
# and the oracle with same-reader examples, mc from plain seed 2, and mc-reader with
# and without same-reader examples. Qwen3.5-27B with the image (518 px).
set -euo pipefail
cd "$(dirname "$0")/.."

s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes "$@"; }
SET=val
MODEL=qwen3.5-27b-vision
SAMPLING="qwen3.5-thinking-32k qwen3.5-thinking-32k-seed2"
SEC=findings-impression
JOINT=findings+impression
POOL=chexpertplus-mimic-rexgradient
PN=joint-naive-negatives-indication-image
PS=joint-simple-negatives-indication-image
dataset=mimic

labels=""
declare -A lab
for d in chexpertplus mimic rexgradient; do
  for section in findings impression; do
    lab[$d.$section]=$(s --gres=gpu:1 -- labrag labels --dataset $d --section $section)
    labels+=":${lab[$d.$section]}"
  done
done
FT="--gres=gpu:4 --cpus-per-task=144 --mem=256G -- labrag finetune --dataset $POOL --section $SEC --multiclass"
ft1=$(s -d afterok$labels $FT --seed 1)
ft2=$(s -d afterok$labels $FT --seed 2)
ftr=$(s -d afterok$labels $FT --seed 1 --classifier attn-reader)
ids="labels ${labels#:} finetune $ft1 $ft2 reader $ftr"

F=val/$dataset/$JOINT
B="labrag build-prompts --experiment val --dataset $dataset --section $JOINT"
lj=$(s -d afterok:${lab[$dataset.findings]}:${lab[$dataset.impression]} --cpus-per-task=8 -- labrag labels --dataset $dataset --section $JOINT)
emb=$(s --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag embed --dataset $dataset)
declare -A prm
for name in ft-attn-seed1:$ft1 ft-attn-seed2:$ft2 ft-attn-reader-seed1:$ftr; do
  clf=$POOL/raddino.${name%%:*}/$SEC/chexbert14.mc
  prm[${name%%:*}]=$(s -d afterok:${name#*:} --gres=gpu:1 --cpus-per-task=32 --mem=64G -- labrag predict --dataset $dataset --classifier $clf)
done
ids+=" labels $lj embed $emb predict ${prm[*]}"

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
mc() { echo "--condition labrag --predictions $POOL/raddino.$1/$SEC/chexbert14.mc --prompt $PS"; }
L=$F/labrag.$POOL
cell retrieval-only.negatives.indication $lj:$emb $F/retrieval-only/$PN --condition retrieval-only --prompt $PN
cell retrieval-only.negatives.indication.same-reader $lj:$emb $F/retrieval-only.same-reader/$PN --condition retrieval-only --prompt $PN --same-reader
cell oracle.negatives.indication $lj:$emb $F/labrag-oracle.chexbert14/$PS --condition labrag-oracle --prompt $PS
cell oracle.negatives.indication.same-reader $lj:$emb $F/labrag-oracle.chexbert14.same-reader/$PS --condition labrag-oracle --prompt $PS --same-reader
cell mc.positives.negatives.indication $lj:$emb:${prm[ft-attn-seed1]} $L.ft-attn-seed1.$SEC.chexbert14.mc/$PS $(mc ft-attn-seed1)
cell mc.positives.negatives.indication.same-reader $lj:$emb:${prm[ft-attn-seed1]} $L.ft-attn-seed1.$SEC.chexbert14.mc.same-reader/$PS $(mc ft-attn-seed1) --same-reader
cell mc-seed2.positives.negatives.indication $lj:$emb:${prm[ft-attn-seed2]} $L.ft-attn-seed2.$SEC.chexbert14.mc/$PS $(mc ft-attn-seed2)
cell mc-reader.positives.negatives.indication $lj:$emb:${prm[ft-attn-reader-seed1]} $L.ft-attn-reader-seed1.$SEC.chexbert14.mc/$PS $(mc ft-attn-reader-seed1)
cell mc-reader.positives.negatives.indication.same-reader $lj:$emb:${prm[ft-attn-reader-seed1]} $L.ft-attn-reader-seed1.$SEC.chexbert14.mc.same-reader/$PS $(mc ft-attn-reader-seed1) --same-reader
# --overwrite: the comparison gained the oracle same-reader cell after the first run
cmp=$(s -d afterok$dep -- labrag compare --generations $cells --out $F/qwen3.5-27b.reader/$SET --overwrite)
ids+=" compare $cmp"
echo "$ids"
