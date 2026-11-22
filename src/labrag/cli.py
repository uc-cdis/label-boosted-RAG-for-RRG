"""Command-line entry point: ``labrag <command> ...``.

Each command calls one function with the options given as keyword arguments; an
omitted option is not passed, so the function's own default applies. Each command's
parser is built by its own ``add_<command>`` function, listed in COMMANDS.
"""

import argparse

from labrag import artifacts, data, serving
from labrag.classifiers import quant
from labrag.data import DATASETS
from labrag.stages import (
    build_prompts,
    classify,
    compare,
    embed,
    evaluate,
    finetune,
    generate,
    labels,
    predict,
    sides,
    threshold,
)

SUITE = ", ".join(evaluate.SUITE)

# Options shared by several stages.
OPTIONS = {
    "--dataset": dict(required=True, choices=sorted(DATASETS)),
    "--section": dict(required=True, choices=data.SECTIONS),
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


def add_splits(sub) -> None:
    help = "redraw a dataset's frozen split files in place (git diff shows any change)"
    p = command(sub, "splits", data.write_splits, help)
    p.add_argument("--dataset", **OPTIONS["--dataset"])


def add_labels(sub) -> None:
    stage(
        sub,
        labels,
        "--dataset",
        "--section",
        "--labeler",
    )


def add_sides(sub) -> None:
    stage(sub, sides, "--dataset", "--section")


def add_embed(sub) -> None:
    p = stage(
        sub,
        embed,
        "--dataset",
        "--encoder",
    )
    p.add_argument(
        "--feature",
        choices=embed.FEATURES,
        help="cls (default) or tokens: CLS + patch tokens, stored as NVFP4",
    )


def add_classify(sub) -> None:
    p = stage(
        sub,
        classify,
        "--dataset",
        "--encoder",
        "--labeler",
        "--uncertain",
    )
    p.add_argument(
        "--section",
        required=True,
        help=f"labels of a section ({', '.join(data.SECTIONS)}), or of several joined "
        f"by {data.JOIN} (attn: one head, a readout per section), e.g. "
        "findings-impression",
    )
    p.add_argument(
        "--classifier",
        choices=classify.CLASSIFIERS,
    )
    p.add_argument(
        "--tokens",
        choices=quant.FORMATS,
        help="attn: storage format the tokens are read in (default fp16)",
    )
    p.add_argument(
        "--seed",
        type=int,
        help="attn: training seed (default 0)",
    )
    p.add_argument(
        "--folds",
        type=int,
        help="attn: also out-of-fold predictions for the train role, K folds by patient",
    )


def add_finetune(sub) -> None:
    p = stage(sub, finetune, "--labeler", "--uncertain")
    p.add_argument(
        "--dataset",
        required=True,
        help=f"training datasets ({', '.join(sorted(DATASETS))}) joined by {data.JOIN}, "
        "e.g. chexpertplus-mimic-rexgradient",
    )
    p.add_argument(
        "--section",
        required=True,
        help=f"sections joined by {data.JOIN}, a readout each, e.g. findings-impression",
    )
    p.add_argument("--classifier", choices=finetune.CLASSIFIERS)
    p.add_argument("--seed", type=int, help="training seed (default 0)")
    p.add_argument(
        "--multiclass",
        action="store_true",
        help="predict CheXbert's four values per label (positive, negative, uncertain, "
        "blank), not positive vs the rest",
    )


def add_threshold(sub) -> None:
    p = stage(sub, threshold)
    p.add_argument(
        "--classifier",
        required=True,
        help="classify artifact under classifiers/, e.g. "
        "chexpertplus/raddino.attn-nvfp4-seed1/findings/chexbert14",
    )
    p.add_argument(
        "--rule",
        required=True,
        choices=threshold.RULES,
        help="threshold rule on val; name suffix "
        + ", ".join(f"{r} {s}" for r, s in threshold.RULES.items()),
    )


def add_predict(sub) -> None:
    p = stage(sub, predict, "--dataset")
    p.add_argument(
        "--classifier",
        required=True,
        help="classify artifact under classifiers/: <dataset>/<name>/<sections>/<labeler>",
    )


def add_build_prompts(sub) -> None:
    p = stage(
        sub,
        build_prompts,
        "--dataset",
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
    p.add_argument(
        "--classifier",
        help="labrag and its diagnostics: classify or threshold artifact under "
        "classifiers/, <dataset>/<name>/<sections>/<labeler>, e.g. "
        "chexpertplus/raddino.attn-nvfp4-seed1/findings-impression/chexbert14",
    )
    defaults = ", ".join(f"{c} {v}" for c, v in build_prompts.DEFAULT_PROMPT.items())
    p.add_argument(
        "--prompt",
        help=f"version in configs/prompts.yaml (default: {defaults})",
    )
    p.add_argument(
        "--corpus-labels",
        choices=("reference", "predicted"),
        help="labrag: corpus labels from the reports (default) or the classifier "
        "(a -oofK classifier, or the --predictions artifact)",
    )
    p.add_argument(
        "--ranking",
        choices=("labels", "image"),
        help="label conditions: rank examples by label overlap, then image (default), "
        "or by image only, as retrieval-only",
    )
    p.add_argument(
        "--sides",
        action="store_true",
        help="labrag-oracle: show each label with its side (sides stage)",
    )
    p.add_argument(
        "--confidence",
        action="store_true",
        help="labrag: show each target label with the classifier's probability",
    )
    p.add_argument(
        "--same-reader",
        action="store_true",
        help="examples only from studies with the target's reader (the radiologist on "
        "the report; datasets with readers)",
    )
    p.add_argument(
        "--predictions",
        help="labrag and diagnostics: target labels from a predict artifact, its "
        "classifier <source>/<name>/<sections>/<labeler> "
        "(e.g. chexpertplus-mimic-rexgradient/raddino.ft-attn-seed1/"
        "findings-impression/chexbert14)",
    )


def add_generate(sub) -> None:
    p = stage(sub, generate)
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


def add_provenance(sub) -> None:
    help = "list every stale artifact under the shared root (read-only)"
    command(sub, "provenance", artifacts.provenance, help)


# In pipeline order, as `labrag --help` lists them.
COMMANDS = (
    add_splits,
    add_labels,
    add_sides,
    add_embed,
    add_classify,
    add_finetune,
    add_threshold,
    add_predict,
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
    func(**kwargs)


if __name__ == "__main__":
    main()
