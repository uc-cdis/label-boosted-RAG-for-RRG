"""Command-line entry point: ``labrag <command> ...``.

Each command calls one function with the options given as keyword arguments; an
omitted option is not passed, so the function's own default applies. Each command's
parser is built by its own ``add_<command>`` function, listed in COMMANDS.
"""

import argparse
import sys
from pathlib import Path

from labrag import artifacts, paths, serving
from labrag.data import DATASETS, chexpertplus
from labrag.stages import (
    build_prompts,
    classify,
    compare,
    embed,
    evaluate,
    generate,
    labels,
)

SUITE = ", ".join(evaluate.SUITE)

# Options shared by several stages.
OPTIONS = {
    "--dataset": dict(required=True, choices=sorted(DATASETS)),
    "--section": dict(required=True, choices=labels.SECTIONS),
    "--encoder": dict(choices=sorted(embed.ENCODERS)),
    "--labeler": dict(choices=labels.LABELERS),
    "--uncertain": dict(
        choices=sorted(classify.POSITIVE),
        help="uncertain labels count as 0 (zeros, the default) or 1 (ones, the preprint)",
    ),
}


def command(sub, name: str, func, help: str) -> argparse.ArgumentParser:
    """Subcommand ``name`` calling ``func`` with the options given."""
    p = sub.add_parser(name, help=help, argument_default=argparse.SUPPRESS)
    p.set_defaults(func=func)
    return p


def stage(sub, module, *options: str) -> argparse.ArgumentParser:
    """Subcommand for a stage module: its name, the first line of its docstring as
    help, ``module.run``, the given OPTIONS and --overwrite."""
    name = module.__name__.rsplit(".", 1)[1].replace("_", "-")
    help = module.__doc__.splitlines()[0].split(": ", 1)[1].rstrip(".")
    p = command(sub, name, module.run, help.replace("%", "%%"))
    for flag in options:
        p.add_argument(flag, **OPTIONS[flag])
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing artifacts",
    )
    return p


def splits_pilot(
    cxr_root: Path = paths.CXR_ROOT,
    out: Path = paths.SPLITS_DIR / chexpertplus.PILOT_SPLIT_FILE,
    overwrite: bool = False,
) -> None:
    """Redraw the CheXpert Plus pilot split and check it against the committed file.

    The committed file is frozen: it is never overwritten unless --overwrite is given.
    """
    drawn = chexpertplus.draw_pilot_split(chexpertplus.load_images(cxr_root))
    text = drawn.to_csv(index=False)
    if out.exists() and not overwrite:
        if out.read_text() == text:
            print(f"OK: redraw matches {out}")
            return
        sys.exit(f"MISMATCH: redraw differs from {out} (file left untouched)")
    out.write_text(text)
    print(f"Wrote {out} ({len(drawn)} patients)")


def add_splits(sub) -> None:
    splits = sub.add_parser("splits", help="frozen split files").add_subparsers(
        dest="split", required=True
    )
    p = command(splits, "pilot", splits_pilot, splits_pilot.__doc__.splitlines()[0])
    p.add_argument("--cxr-root", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--overwrite", action="store_true", help="replace the frozen file")


def add_labels(sub) -> None:
    stage(
        sub,
        labels,
        "--dataset",
        "--section",
        "--labeler",
    )


def add_embed(sub) -> None:
    stage(
        sub,
        embed,
        "--dataset",
        "--encoder",
    )


def add_classify(sub) -> None:
    p = stage(
        sub,
        classify,
        "--dataset",
        "--section",
        "--encoder",
        "--labeler",
        "--uncertain",
    )
    p.add_argument(
        "--classifier",
        choices=classify.CLASSIFIERS,
    )


def add_build_prompts(sub) -> None:
    p = stage(
        sub,
        build_prompts,
        "--section",
        "--uncertain",
    )
    p.add_argument(
        "--experiment",
        required=True,
        choices=sorted(build_prompts.EXPERIMENTS),
    )
    p.add_argument(
        "--condition",
        required=True,
        choices=build_prompts.CONDITIONS,
    )
    defaults = ", ".join(f"{c} {v}" for c, v in build_prompts.DEFAULT_PROMPT.items())
    p.add_argument(
        "--prompt",
        help=f"version in configs/prompts.yaml (default: {defaults})",
    )


def add_generate(sub) -> None:
    p = stage(
        sub,
        generate,
    )
    p.add_argument(
        "--prompts",
        required=True,
        nargs="+",
        help="prompt artifacts, relative to prompts/ (one server serves them all)",
    )
    p.add_argument(
        "--set",
        dest="target_set",
        metavar="SET",
        required=True,
        help="target set, e.g. pilot",
    )
    p.add_argument(
        "--model",
        required=True,
        choices=sorted(serving.load_config("models")),
    )
    p.add_argument(
        "--sampling",
        required=True,
        choices=sorted(serving.load_config("sampling")),
    )


def add_evaluate(sub) -> None:
    p = stage(sub, evaluate)
    p.add_argument(
        "--generations",
        required=True,
        nargs="+",
        help="generations artifacts, relative to generations/",
    )
    p.add_argument(
        "--metric",
        dest="metrics",
        nargs="+",
        choices=sorted(evaluate.METRICS),
        help=f"metrics (default: the suite, {SUITE}, at once)",
    )


def add_compare(sub) -> None:
    p = stage(sub, compare)
    p.add_argument(
        "--generations",
        required=True,
        nargs="+",
        help="two or more generations artifacts (the conditions), scored on the same "
        "studies, relative to generations/",
    )
    p.add_argument(
        "--out",
        required=True,
        help="output artifact, relative to comparisons/",
    )
    p.add_argument(
        "--metric",
        dest="metrics",
        nargs="+",
        choices=sorted(evaluate.METRICS),
        help=f"metrics (default: the suite, {SUITE})",
    )


def provenance() -> None:
    """List every stale artifact under the shared root (read-only)."""
    seen, stale = artifacts.stale_tree()
    for path, reasons in sorted(stale.items()):
        print(f"STALE {path.relative_to(paths.SHARED_ROOT)}")
        for reason in reasons:
            print(f"  - {reason}")
    print(f"{seen} artifacts, {len(stale)} stale")
    if stale:
        sys.exit(1)


def add_provenance(sub) -> None:
    command(sub, "provenance", provenance, provenance.__doc__)


# In pipeline order, as `labrag --help` lists them.
COMMANDS = (
    add_splits,
    add_labels,
    add_embed,
    add_classify,
    add_build_prompts,
    add_generate,
    add_evaluate,
    add_compare,
    add_provenance,
)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="labrag")
    sub = ap.add_subparsers(dest="command", required=True)
    for add in COMMANDS:
        add(sub)

    kwargs = vars(ap.parse_args(argv))
    func = kwargs.pop("func")
    del kwargs["command"]
    kwargs.pop("split", None)
    func(**kwargs)


if __name__ == "__main__":
    main()
