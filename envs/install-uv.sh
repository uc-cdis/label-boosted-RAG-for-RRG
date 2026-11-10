#!/usr/bin/env bash
# Install the pinned uv release if it isn't already on this host.
#
#   envs/install-uv.sh            # installs to ~/.local/bin (override with UV_INSTALL_DIR)
#
# Downloads the official GitHub release tarball and checks it against the sha256 pinned
# here, not against a checksum fetched alongside it. Does nothing if `uv` in the install
# dir already reports the pinned version. Prints the path of the uv binary.
set -euo pipefail

version=0.13.7
target=uv-x86_64-unknown-linux-gnu
sha256=c5221ff41b7892bfc223543883600125f6a47fe845093e0f5bdff4f359196d83

dest=${UV_INSTALL_DIR:-$HOME/.local/bin}
if [[ -x $dest/uv ]] && [[ $("$dest/uv" --version) == "uv $version "* ]]; then
  echo "$dest/uv"
  exit 0
fi
[[ $(uname -sm) == "Linux x86_64" ]] || { echo "unsupported platform: $(uname -sm)" >&2; exit 1; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -sSfL -o "$tmp/uv.tar.gz" \
  "https://github.com/astral-sh/uv/releases/download/$version/$target.tar.gz"
echo "$sha256  $tmp/uv.tar.gz" | sha256sum --check --quiet
tar -xzf "$tmp/uv.tar.gz" -C "$tmp"
mkdir -p "$dest"
install -m 755 "$tmp/$target/uv" "$tmp/$target/uvx" "$dest/"
echo "$dest/uv"
