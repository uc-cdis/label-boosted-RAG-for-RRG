#!/usr/bin/env bash
# Move locally derived files (anything not released by the dataset authors) into a
# `derived/` subfolder of the dataset they came from. The list comes from the 2026-11-10
# audit (docs/data.md). Dry run by default: prints the moves. Idempotent: moves only what
# is still in place, never overwrites (mv -n), and fails if a target already exists.
#
#   tools/data-audit/segregate.sh [CXR_ROOT] [--apply]
set -euo pipefail
root=/opt/gpudata/cxr; apply=0
for a in "$@"; do
  case $a in --apply) apply=1 ;; *) root=$a ;; esac
done
cd "$root"

moves=()   # "src<TAB>dst_dir"
add() { local dst=$1; shift; for src in "$@"; do moves+=("$src"$'\t'"$dst"); done; }

# Top-level derived/: old section/label csvs, named by dataset.
add chexpertplus/derived derived/chexpertplus-*
add mimic-cxr/derived     derived/mimic-*
add openi/derived         derived/openi-*
# Mixed into the dataset dirs.
add mimic-cxr/derived     mimic-cxr/mimic_cxr_sectioned.csv mimic-cxr/mimic_cxr_selected_section.csv \
                          mimic-cxr/chexbert_labels.csv
add chexpertplus/derived  chexpertplus/split.csv chexpertplus/metadata.csv chexpertplus/report.csv
add rexgradient/derived   rexgradient/metadata/train_chexbert_labels.csv \
                          rexgradient/metadata/valid_chexbert_labels.csv \
                          rexgradient/metadata/test_chexbert_labels.csv rexgradient/by-site
add openi/derived         openi/metadata.csv openi/report.csv openi/split.csv openi/IU-xray-projections.csv

n=0
for m in "${moves[@]}"; do
  src=${m%%$'\t'*}; dst=${m#*$'\t'}
  [[ -e $src ]] || continue            # already moved, or the glob matched nothing
  [[ -e $dst/$(basename "$src") ]] && { echo "target exists: $dst/$(basename "$src")" >&2; exit 1; }
  echo "mv $src $dst/"; n=$((n + 1))
  if (( apply )); then mkdir -p "$dst"; mv -n "$src" "$dst/"; fi
done
if (( apply )); then
  [[ -d derived ]] && rmdir derived 2>/dev/null && echo "rmdir derived"
  echo "moved $n"
else
  echo "$n moves (dry run; add --apply)"
fi
