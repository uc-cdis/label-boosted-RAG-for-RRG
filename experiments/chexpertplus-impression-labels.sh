#!/usr/bin/env bash
# CheXbert-14 labels on CheXpert Plus Impression text, for the Findings vs Impression
# label agreement that decides the full-dataset classifier's label targets.
set -euo pipefail
cd "$(dirname "$0")/.."

slurm/submit.sh --gres=gpu:1 -- labrag labels --dataset chexpertplus --section impression
