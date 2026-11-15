"""Each command passes its options as keyword arguments its function accepts."""

import inspect

import pytest

from labrag import cli
from labrag.stages import build_prompts, classify, compare, embed, generate, labels

CASES = [
    (
        labels,
        "labels --dataset chexpertplus --section findings",
        dict(dataset="chexpertplus", section="findings"),
    ),
    (
        embed,
        "embed --dataset chexpertplus --encoder raddino --overwrite",
        dict(dataset="chexpertplus", encoder="raddino", overwrite=True),
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


def test_evaluate_suite_and_repro(monkeypatch):
    calls = []
    monkeypatch.setattr(cli.evaluate, "run", lambda *a: calls.append(("run", *a)))
    monkeypatch.setattr(cli.evaluate, "repro", lambda m: calls.append(("repro", m)))
    cli.main("evaluate --generations a b".split())
    cli.main("evaluate --generations a --metric green --overwrite".split())
    cli.main("evaluate --repro --metric green".split())
    assert calls == [
        ("run", ["a", "b"], None, False),
        ("run", ["a"], ["green"], True),
        ("repro", "green"),
    ]
    with pytest.raises(SystemExit):
        cli.main("evaluate --repro --metric green --generations a".split())
