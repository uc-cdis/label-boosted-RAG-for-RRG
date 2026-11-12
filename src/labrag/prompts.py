"""Prompt versions (configs/prompts.yaml) and rendering them into chat messages."""

import string

import yaml

from labrag import paths

PROMPTS_YAML = paths.CONFIGS_DIR / "prompts.yaml"
NO_POSITIVE = "Other"  # the preprint's implicit label for "no positive label"


def load(name: str) -> dict:
    """One prompt version: {"example": str, "user": str}."""
    versions = yaml.safe_load(PROMPTS_YAML.read_text())
    if name not in versions:
        raise SystemExit(f"unknown prompt {name!r}; known: {sorted(versions)}")
    version = versions[name]
    assert set(version) == {"example", "user"}, f"{name}: unexpected fields"
    fields = {f for _, f, _, _ in string.Formatter().parse(version["user"]) if f}
    assert {"examples"} <= fields <= {"examples", "labels"}, f"{name}: {fields}"
    return version


def uses_labels(version: dict) -> bool:
    return "{labels}" in version["user"]


def clean_report(text: str) -> str:
    """Report text as shown in a prompt: every whitespace run becomes one space.

    CheXpert Plus Findings carry fixed-width line wraps, which the model would copy.
    The reference text that generations are scored against is not changed.
    """
    return " ".join(text.split())


def label_text(names: list[str]) -> str:
    return ", ".join(names) if names else NO_POSITIVE


def render(
    version: dict,
    examples: list[tuple[str, list[str]]],
    labels: list[str] | None,
) -> list[dict]:
    """Chat messages for one target.

    ``examples``: (report text, positive label names) per retrieved study, best first.
    ``labels``: the target's positive label names (None for a version without labels).
    """
    with_labels = uses_labels(version)
    assert with_labels == (labels is not None), "labels given iff the version uses them"
    blocks = [
        version["example"].format(
            index=i,
            report=clean_report(report),
            **({"labels": label_text(names)} if with_labels else {}),
        )
        for i, (report, names) in enumerate(examples, start=1)
    ]
    fields = {"examples": "\n\n".join(blocks)}
    if with_labels:
        fields["labels"] = label_text(labels)
    return [{"role": "user", "content": version["user"].format(**fields).strip()}]
