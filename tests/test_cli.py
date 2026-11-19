"""Each command passes its options as keyword arguments its function accepts."""

import inspect

import pytest

from labrag import cli
from labrag.stages import (
    build_prompts,
    classify,
    compare,
    embed,
    evaluate,
    generate,
    labels,
    predict,
    threshold,
)

CASES = [
    (
        labels,
        "labels --dataset chexpertplus --section findings",
        dict(dataset="chexpertplus", section="findings"),
    ),
    (
        labels,
        "labels --dataset chexpertplus --section impression",
        dict(dataset="chexpertplus", section="impression"),
    ),
    (
        embed,
        "embed --dataset chexpertplus --encoder raddino --overwrite",
        dict(dataset="chexpertplus", encoder="raddino", overwrite=True),
    ),
    (
        classify,
        "classify --dataset chexpertplus --section findings --classifier attn "
        "--tokens nvfp4 --seed 2",
        dict(
            dataset="chexpertplus",
            section="findings",
            classifier="attn",
            tokens="nvfp4",
            seed=2,
        ),
    ),
    (
        classify,
        "classify --dataset chexpertplus --section findings --classifier attn-cls "
        "--tokens nvfp4 --seed 1",
        dict(
            dataset="chexpertplus",
            section="findings",
            classifier="attn-cls",
            tokens="nvfp4",
            seed=1,
        ),
    ),
    (
        classify,
        "classify --dataset chexpertplus --section findings --uncertain ones "
        "--classifier lr --labeler chexbert14",
        dict(
            dataset="chexpertplus",
            section="findings",
            uncertain="ones",
            classifier="lr",
            labeler="chexbert14",
        ),
    ),
    (
        labels,
        "labels --dataset padchest --section report --labeler padchest60",
        dict(dataset="padchest", section="report", labeler="padchest60"),
    ),
    (
        embed,
        "embed --dataset chexpertplus --feature tokens",
        dict(dataset="chexpertplus", feature="tokens"),
    ),
    (
        classify,
        "classify --dataset chexpertplus --section findings-impression "
        "--classifier attn --tokens nvfp4 --seed 1",
        dict(
            dataset="chexpertplus",
            section="findings-impression",
            classifier="attn",
            tokens="nvfp4",
            seed=1,
        ),
    ),
    (
        embed,
        "embed --dataset padchest --feature tokens",
        dict(dataset="padchest", feature="tokens"),
    ),
    (
        classify,
        "classify --dataset padchest --section report --labeler padchest60 "
        "--classifier attn --tokens nvfp4 --seed 1",
        dict(
            dataset="padchest",
            section="report",
            labeler="padchest60",
            classifier="attn",
            tokens="nvfp4",
            seed=1,
        ),
    ),
    (
        classify,
        "classify --dataset chexpertplus --section findings --classifier attn "
        "--tokens nvfp4 --seed 1 --folds 5",
        dict(
            dataset="chexpertplus",
            section="findings",
            classifier="attn",
            tokens="nvfp4",
            seed=1,
            folds=5,
        ),
    ),
    (
        threshold,
        "threshold --classifier chexpertplus/raddino.attn-nvfp4-seed1/"
        "findings-impression/chexbert14 --rule prevalence",
        dict(
            classifier="chexpertplus/raddino.attn-nvfp4-seed1/findings-impression/"
            "chexbert14",
            rule="prevalence",
        ),
    ),
    (
        predict,
        "predict --dataset chexpertplus "
        "--classifier padchest/raddino.attn-nvfp4-seed1/report/padchest60",
        dict(
            dataset="chexpertplus",
            classifier="padchest/raddino.attn-nvfp4-seed1/report/padchest60",
        ),
    ),
    (
        build_prompts,
        "build-prompts --experiment pilot --section findings --condition labrag "
        "--prompt preprint-simple",
        dict(
            experiment="pilot",
            section="findings",
            condition="labrag",
            prompt="preprint-simple",
        ),
    ),
    (
        build_prompts,
        "build-prompts --experiment pilot --section findings --condition labrag "
        "--classifier chexpertplus/raddino.attn-nvfp4-seed1/findings/chexbert14",
        dict(
            experiment="pilot",
            section="findings",
            condition="labrag",
            classifier="chexpertplus/raddino.attn-nvfp4-seed1/findings/chexbert14",
        ),
    ),
    (
        build_prompts,
        "build-prompts --experiment pilot --section findings --condition labrag "
        "--classifier chexpertplus/raddino.attn-nvfp4-seed1-oof5/findings/chexbert14 "
        "--corpus-labels predicted",
        dict(
            experiment="pilot",
            section="findings",
            condition="labrag",
            classifier="chexpertplus/raddino.attn-nvfp4-seed1-oof5/findings/chexbert14",
            corpus_labels="predicted",
        ),
    ),
    (
        build_prompts,
        "build-prompts --experiment pilot --section findings --condition labrag "
        "--predictions padchest/raddino.attn-nvfp4-seed1/report/padchest60 "
        "--corpus-labels predicted",
        dict(
            experiment="pilot",
            section="findings",
            condition="labrag",
            predictions="padchest/raddino.attn-nvfp4-seed1/report/padchest60",
            corpus_labels="predicted",
        ),
    ),
    (
        generate,
        "generate --prompts a b --set pilot --model qwen3.5-27b "
        "--sampling qwen3.5-thinking-32k",
        dict(
            prompts=["a", "b"],
            target_set="pilot",
            model="qwen3.5-27b",
            sampling="qwen3.5-thinking-32k",
        ),
    ),
    (
        evaluate,
        "evaluate --generations a b",
        dict(generations=["a", "b"]),
    ),
    (
        evaluate,
        "evaluate --generations a --metric green ratescore --overwrite",
        dict(generations=["a"], metrics=["green", "ratescore"], overwrite=True),
    ),
    (
        compare,
        "compare --generations a b --out c --metric green ratescore",
        dict(generations=["a", "b"], out="c", metrics=["green", "ratescore"]),
    ),
]


@pytest.mark.parametrize("module, argv, expected", CASES)
def test_stage_options_reach_run(monkeypatch, module, argv, expected):
    calls = []

    def record(**kwargs):
        inspect.signature(module.run).bind(**kwargs)  # TypeError on a stray option
        calls.append(kwargs)

    monkeypatch.setattr(module, "run", record)
    cli.main(argv.split())
    assert calls == [expected]


def test_splits_reach_write_splits(monkeypatch):
    calls = []
    monkeypatch.setattr(cli.data, "write_splits", lambda **kw: calls.append(kw))
    cli.main("splits --dataset chexpertplus".split())
    assert calls == [dict(dataset="chexpertplus")]
