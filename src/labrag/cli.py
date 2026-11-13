"""Command-line entry point: ``labrag <command> ...``."""

import argparse
import sys
from pathlib import Path

from labrag import paths, serving
from labrag.data import DATASETS, chexpertplus
from labrag.stages import build_prompts, classify, embed, evaluate, generate, labels

PILOT_SPLIT_FILE = chexpertplus.PILOT_SPLIT_FILE


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

    p = sub.add_parser("classify", help="per-label image classifiers (stage classify)")
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--section", required=True, choices=labels.SECTIONS)
    p.add_argument("--encoder", default="raddino", choices=sorted(embed.ENCODERS))
    p.add_argument("--labeler", default="chexbert14", choices=labels.LABELERS)
    p.add_argument("--classifier", default="lr", choices=classify.CLASSIFIERS)
    p.add_argument(
        "--uncertain",
        default="ones",
        choices=sorted(classify.POSITIVE),
        help="uncertain labels count as 1 (ones, the preprint) or 0 (zeros)",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="replace an existing artifact"
    )
    p.set_defaults(
        func=lambda a: classify.run(
            a.dataset,
            a.section,
            a.encoder,
            a.labeler,
            a.classifier,
            a.uncertain,
            a.overwrite,
        )
    )

    p = sub.add_parser(
        "build-prompts", help="one chat prompt per target study (stage build-prompts)"
    )
    p.add_argument(
        "--experiment", required=True, choices=sorted(build_prompts.EXPERIMENTS)
    )
    p.add_argument("--section", required=True, choices=labels.SECTIONS)
    p.add_argument("--condition", required=True, choices=build_prompts.CONDITIONS)
    p.add_argument(
        "--prompt",
        help="version in configs/prompts.yaml (default: "
        + ", ".join(f"{c} {v}" for c, v in build_prompts.DEFAULT_PROMPT.items())
        + ")",
    )
    p.add_argument(
        "--uncertain",
        default="ones",
        choices=sorted(classify.POSITIVE),
        help="labrag only: which classifier variant (as in classify)",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="replace an existing artifact"
    )
    p.set_defaults(
        func=lambda a: build_prompts.run(
            a.experiment, a.section, a.condition, a.prompt, a.uncertain, a.overwrite
        )
    )

    p = sub.add_parser(
        "generate", help="generations from a vLLM-served model (stage generate)"
    )
    p.add_argument(
        "--prompts",
        required=True,
        nargs="+",
        help="prompt artifacts, relative to prompts/ (one server serves them all)",
    )
    p.add_argument("--set", required=True, help="target set, e.g. pilot")
    p.add_argument(
        "--model", required=True, choices=sorted(serving.load_config("models"))
    )
    p.add_argument(
        "--sampling", required=True, choices=sorted(serving.load_config("sampling"))
    )
    p.add_argument(
        "--overwrite", action="store_true", help="replace existing artifacts"
    )
    p.set_defaults(
        func=lambda a: generate.run(a.prompts, a.set, a.model, a.sampling, a.overwrite)
    )

    p = sub.add_parser(
        "evaluate", help="per-study scores by official metric code (stage evaluate)"
    )
    p.add_argument(
        "--generations",
        required=True,
        nargs="+",
        help="generations artifacts, relative to generations/",
    )
    p.add_argument("--metric", required=True, choices=sorted(evaluate.METRICS))
    p.add_argument(
        "--overwrite", action="store_true", help="replace existing artifacts"
    )
    p.set_defaults(func=lambda a: evaluate.run(a.generations, a.metric, a.overwrite))

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
