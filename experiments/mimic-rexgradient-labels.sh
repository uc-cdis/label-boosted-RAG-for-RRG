#!/usr/bin/env bash
# CheXbert-14 labels on MIMIC-CXR and ReXGradient Findings and Impression: the in-domain
# references, and with CheXpert Plus the training labels of the pooled classifier.
set -euo pipefail
cd "$(dirname "$0")/.."

for dataset in mimic rexgradient; do
  for section in findings impression; do
    slurm/submit.sh --gres=gpu:1 -- labrag labels --dataset $dataset --section $section
  done
done
