#!/usr/bin/env bash
# Install the Hugging Face CLI (`hf`) as a uv tool, at the versions in this commit's uv.lock.
#
#   envs/install-hf.sh     # tool env in $UV_TOOL_DIR, `hf` in $UV_TOOL_BIN_DIR
#
# For pre-pulling model weights by hand (`hf download <repo> --local-dir ...`, see
# configs/weights.yaml); jobs never download. huggingface_hub and all its dependencies are
# pinned to uv.lock, so `hf` writes the same download metadata as the pipeline env reads.
# Does nothing if `hf` in UV_TOOL_BIN_DIR already reports the locked version. Uses the
# pinned uv (install-uv.sh) and the Python in .python-version. Prints the path of `hf`.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bin=${UV_TOOL_BIN_DIR:?set UV_TOOL_BIN_DIR (and UV_TOOL_DIR) to where the tool goes}
: "${UV_TOOL_DIR:?set UV_TOOL_DIR}"

uv=$("$repo/envs/install-uv.sh")
locked=$(cd "$repo" && "$uv" export --frozen --no-dev --no-hashes --no-emit-project \
  --no-header --no-annotate)
version=$(sed -n 's/^huggingface-hub==\([^ ;]*\).*/\1/p' <<<"$locked")
[[ -n $version ]] || { echo "huggingface-hub not found in uv.lock" >&2; exit 1; }

if [[ -x $bin/hf ]] && [[ $("$bin/hf" version 2>/dev/null) == "version=$version" ]]; then
  echo "$bin/hf"
  exit 0
fi
constraints=$(mktemp)
trap 'rm -f "$constraints"' EXIT
echo "$locked" >"$constraints"
"$uv" tool install --force --managed-python --python "$(cat "$repo/.python-version")" \
  --constraints "$constraints" "huggingface-hub==$version" >&2
echo "$bin/hf"
