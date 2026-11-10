#!/usr/bin/env bash
# Hash every file of every dataset under a CXR root into one sha256sum-format manifest per
# dataset, sorted by path, so manifests from different nodes can be diffed directly.
#
#   tools/data-audit/manifest.sh [CXR_ROOT] OUT_DIR
#
# CXR_ROOT defaults to /opt/gpudata/cxr. Skips `<dataset>/derived/` (local derivations, see
# segregate.sh) and a top-level `derived/`. Needs only bash, GNU find/xargs/sort and
# sha256sum, so it runs on any node without the conda env. About 3 GB/s with 32 workers.
set -euo pipefail
if [[ $# -eq 1 ]]; then root=/opt/gpudata/cxr; out=$1; else root=${1:?}; out=${2:?}; fi
out=$(mkdir -p "$out" && cd "$out" && pwd)
jobs=${JOBS:-32}
for dir in "$root"/*/; do
  ds=$(basename "$dir")
  [[ $ds == derived ]] && continue
  t=$SECONDS
  (cd "$dir" && find . -path ./derived -prune -o -type f -printf '%P\0' \
     | sort -z | ionice -c3 xargs -0 -r -P "$jobs" -n 500 sha256sum) \
    | LC_ALL=C sort -k2 > "$out/$ds.sha256.tmp"
  mv "$out/$ds.sha256.tmp" "$out/$ds.sha256"
  echo "$ds: $(wc -l < "$out/$ds.sha256") files, $((SECONDS - t)) s"
done
(cd "$out" && sha256sum -- *.sha256 > digests.txt)
echo "wrote $out/*.sha256 and $out/digests.txt"
