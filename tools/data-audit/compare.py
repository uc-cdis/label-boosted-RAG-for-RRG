#!/usr/bin/env python3
"""Compare a node's dataset manifests (from manifest.sh) against the reference node's.

    python3 tools/data-audit/compare.py REFERENCE_DIR NODE_DIR

Both dirs hold one `<dataset>.sha256` per dataset. Prints, per dataset, the files that are
missing on the node, extra on the node, or differ in content. Exit status 1 if anything
differs. Standard library only, so it runs with any python3 >= 3.8 on the host.
"""
import sys
from pathlib import Path


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            digest, name = line.rstrip("\n").split("  ", 1)
            out[name] = digest
    return out


def main(ref_dir, node_dir, show=10):
    ref_dir, node_dir = Path(ref_dir), Path(node_dir)
    ref = {p.name[: -len(".sha256")] for p in ref_dir.glob("*.sha256")}
    node = {p.name[: -len(".sha256")] for p in node_dir.glob("*.sha256")}
    bad = False
    for ds in sorted(ref | node):
        if ds not in node or ds not in ref:
            print(f"{ds}: only on {'reference' if ds in ref else 'node'}")
            bad = True
            continue
        a, b = load(ref_dir / f"{ds}.sha256"), load(node_dir / f"{ds}.sha256")
        missing = sorted(a.keys() - b.keys())
        extra = sorted(b.keys() - a.keys())
        differ = sorted(k for k in a.keys() & b.keys() if a[k] != b[k])
        if not (missing or extra or differ):
            print(f"{ds}: identical ({len(a)} files)")
            continue
        bad = True
        print(
            f"{ds}: {len(missing)} missing, {len(extra)} extra, {len(differ)} differ (of {len(a)})"
        )
        for label, names in (
            ("missing", missing),
            ("extra", extra),
            ("differ", differ),
        ):
            for name in names[:show]:
                print(f"  {label}: {name}")
            if len(names) > show:
                print(f"  {label}: ... {len(names) - show} more")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
