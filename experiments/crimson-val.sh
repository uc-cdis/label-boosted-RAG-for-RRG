#!/usr/bin/env bash
# CRIMSON (served judge, deterministic settings) added to the val results of the
# methods draft: incidental-readouts-val's cells on MIMIC-CXR and ReXGradient (2
# sampling seeds) and reader-val's matched-radiologist retrieval on MIMIC-CXR. One
# evaluate job per generations artifact (CRIMSON only; the other metrics are built),
# then the three comparisons rebuilt on the whole suite. (The two MIMIC sets the
# crimson-check scored with the earlier judge flags were overwritten by 7c32dcc's run.)
set -euo pipefail
cd "$(dirname "$0")/.."

s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes "$@"; }
SET=val
MODEL=qwen3.5-27b-vision
SAMPLING="qwen3.5-thinking-32k qwen3.5-thinking-32k-seed2"
SEC=findings-impression
JOINT=findings+impression
POOL=chexpertplus-mimic-rexgradient
MDL=labrag.$POOL.ft-attn-seed1.$SEC.chexbert14.devices.mc
MIL=labrag.$POOL.ft-attn-seed1.$SEC.chexbert14.devices.incidentals.mc
ORD=labrag-oracle.chexbert14.devices
PNC=joint-naive-negatives-indication-prior-changes-image
PSC=joint-simple-negatives-indication-prior-changes-image
PN=joint-naive-negatives-indication-image
PS=joint-simple-negatives-indication-image
MC=labrag.$POOL.ft-attn-seed1.$SEC.chexbert14.mc
MC2=labrag.$POOL.ft-attn-seed2.$SEC.chexbert14.mc
MCR=labrag.$POOL.ft-attn-reader-seed1.$SEC.chexbert14.mc
OR=labrag-oracle.chexbert14

ids=""
# comparison <dataset> <name> <condition=prompt dir>...: CRIMSON per seed, then compare
comparison() {
  local dataset=$1 name=$2 F=val/$1/$JOINT
  shift 2
  local cells="" dep="" item cond dir sampling g e
  for item in "$@"; do
    cond=${item%%=*} dir=${item#*=}
    local runs=""
    for sampling in $SAMPLING; do
      g=$F/$dir/$MODEL.$sampling/$SET
      e=$(s --gres=gpu:1 --cpus-per-task=16 -- labrag evaluate --metric crimson --generations $g)
      ids+=" $e"
      dep+=":$e"
      runs+=",$g"
    done
    cells+=" $cond=${runs#,}"
  done
  ids+=" compare $(s -d afterok$dep -- labrag compare --generations $cells --out $F/$name/$SET --overwrite)"
}

for dataset in mimic rexgradient; do
  ids+=" $dataset:"
  comparison $dataset qwen3.5-27b.incidental-readouts \
    retrieval-only.negatives.indication.prior-changes=retrieval-only/$PNC \
    mc-devices.positives.negatives.indication.prior-changes.devices=$MDL.devices/$PSC \
    oracle.negatives.indication.prior-changes.devices=$ORD/$PSC \
    oracle.negatives.indication.prior-changes.devices.incidentals=$ORD.incidentals/$PSC \
    mc-incidentals.positives.negatives.indication.prior-changes.devices=$MIL.devices/$PSC \
    mc-incidentals.positives.negatives.indication.prior-changes.devices.incidentals=$MIL.devices.incidentals/$PSC
done
ids+=" mimic reader:"
comparison mimic qwen3.5-27b.reader \
  retrieval-only.negatives.indication=retrieval-only/$PN \
  retrieval-only.negatives.indication.same-reader=retrieval-only.same-reader/$PN \
  oracle.negatives.indication=$OR/$PS \
  oracle.negatives.indication.same-reader=$OR.same-reader/$PS \
  mc.positives.negatives.indication=$MC/$PS \
  mc.positives.negatives.indication.same-reader=$MC.same-reader/$PS \
  mc-seed2.positives.negatives.indication=$MC2/$PS \
  mc-reader.positives.negatives.indication=$MCR/$PS \
  mc-reader.positives.negatives.indication.same-reader=$MCR.same-reader/$PS
echo "$ids"
