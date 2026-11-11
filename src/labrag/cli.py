"""Command-line entry point: ``labrag <command> ...``."""

import argparse
import sys
from pathlib import Path

from labrag import paths
from labrag.data import DATASETS, chexpertplus
from labrag.stages import embed, labels

PILOT_SPLIT_FILE = "chexpertplus_pilot_split.csv"


def splits_pilot(args: argparse.Namespace) -> None:
    """Redraw the CheXpert Plus pilot split and check it against the committed file.

    The committed file is frozen: it is never overwritten unless --overwrite is given.
    """
    out = args.out
    drawn = chexpertplus.draw_pilot_split(chexpertplus.load_images(args.cxr_root))
    text = drawn.to_csv(index=False)
    if out.exists() and not args.overwrite:
        if out.read_text() == text:
            print(f"OK: redraw matches {out}")
            return
        sys.exit(f"MISMATCH: redraw differs from {out} (file left untouched)")
    out.write_text(text)
    print(f"Wrote {out} ({len(drawn)} patients)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="labrag")
    sub = ap.add_subparsers(dest="command", required=True)

    splits = sub.add_parser("splits", help="frozen split files").add_subparsers(
        dest="split", required=True
    )
    p = splits.add_parser("pilot", help=splits_pilot.__doc__.splitlines()[0])
    p.add_argument("--cxr-root", type=Path, default=paths.CXR_ROOT)
    p.add_argument("--out", type=Path, default=paths.SPLITS_DIR / PILOT_SPLIT_FILE)
    p.add_argument("--overwrite", action="store_true", help="replace the frozen file")
    p.set_defaults(func=splits_pilot)

    p = sub.add_parser("labels", help="reference labels on report text (stage labels)")
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--section", required=True, choices=labels.SECTIONS)
    p.add_argument("--labeler", default="chexbert14", choices=labels.LABELERS)
    p.add_argument(
        "--overwrite", action="store_true", help="replace an existing artifact"
    )
    p.set_defaults(
        func=lambda a: labels.run(a.dataset, a.section, a.labeler, a.overwrite)
    )

    p = sub.add_parser("embed", help="one image embedding per study (stage embed)")
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--encoder", default="raddino", choices=sorted(embed.ENCODERS))
    p.add_argument(
        "--overwrite", action="store_true", help="replace an existing artifact"
    )
    p.set_defaults(func=lambda a: embed.run(a.dataset, a.encoder, a.overwrite))

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
