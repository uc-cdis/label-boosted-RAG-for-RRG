"""Prompt versions (configs/prompts.yaml) and rendering them into chat messages."""

import string

import yaml

from labrag import paths

PROMPTS_YAML = paths.CONFIGS_DIR / "prompts.yaml"
NO_POSITIVE = "Other"  # the preprint's implicit label for "no positive label"
# Where {image} was: generate swaps in the study's image (the record's "image" path).
IMAGE_PART = {"type": "image"}
_IMAGE_MARK = "\x00image\x00"  # never in report text
# {section}: the name of the section generated (Findings+Impression needs its own
# wording: the model writes both headers).
SECTION_NAMES = {"findings": "Findings", "impression": "Impression"}


# label_categories: [...] -> each label under its CheXbert value (1, 0, -1, blank), one
# line per category listed (in that order); labels of the other categories aren't shown
CATEGORIES = {1.0: "Positive", 0.0: "Negative", -1.0: "Uncertain", None: "Unmentioned"}


def load(name: str) -> dict:
    """One prompt version: {"user": str} plus "example" iff it shows examples, and
    optionally "label_categories" (a list of CATEGORIES names)."""
    versions = yaml.safe_load(PROMPTS_YAML.read_text())
    if name not in versions:
        raise SystemExit(f"unknown prompt {name!r}; known: {sorted(versions)}")
    version = versions[name]
    fields = {f for _, f, _, _ in string.Formatter().parse(version["user"]) if f}
    assert fields <= {
        "examples",
        "labels",
        "absent",
        "image",
        "section",
    }, f"{name}: {fields}"
    assert fields & {"examples", "image"}, f"{name}: no examples and no image"
    expected = {"user", "example"} if "examples" in fields else {"user"}
    assert set(version) - {"label_categories"} == expected, f"{name}: {sorted(version)}"
    shown = version.get("label_categories", [])
    assert set(shown) <= set(CATEGORIES.values()) and len(set(shown)) == len(
        shown
    ), name
    assert not shown or "labels" in fields, name
    assert "absent" not in fields or ("labels" in fields and not shown), name
    assert version["user"].count("{image}") <= 1, f"{name}: more than one image"
    return version


def uses_labels(version: dict) -> bool:
    return "{labels}" in version["user"]


def uses_examples(version: dict) -> bool:
    return "{examples}" in version["user"]


def uses_image(version: dict) -> bool:
    return "{image}" in version["user"]


def uses_absent(version: dict) -> bool:
    """{absent}: the target's labels predicted absent (the multiclass negative class),
    next to its positives; the examples show positives only."""
    return "{absent}" in version["user"]


def uses_categories(version: dict) -> bool:
    return "label_categories" in version


def categories(codes, names: list[str], shown: list[str]) -> dict[str, list[str]]:
    """Label names under their CATEGORIES (CheXbert values; NaN = blank), for the
    ``shown`` categories only, in that order."""
    out = {c: [] for c in shown}
    for name, code in zip(names, codes):
        key = None if code is None or code != code else float(code)
        if CATEGORIES[key] in out:
            out[CATEGORIES[key]].append(name)
    return out


def uses_section(version: dict) -> bool:
    return "{section}" in version["user"]


def text_chars(messages: list[dict]) -> int:
    """Characters of text in the messages (image parts not counted)."""
    n = 0
    for m in messages:
        parts = m["content"]
        if isinstance(parts, str):
            n += len(parts)
        else:
            n += sum(len(p["text"]) for p in parts if p["type"] == "text")
    return n


def clean_report(text: str) -> str:
    """Report text as shown in a prompt: every whitespace run becomes one space.

    CheXpert Plus Findings carry fixed-width line wraps, which the model would copy.
    The reference text that generations are scored against is not changed.
    """
    return " ".join(text.split())


def with_probability(name: str, p: float) -> str:
    """A predicted label with its probability: "Atelectasis (28%)". Clamped to 1-99%:
    a classifier is never certain."""
    return f"{name} ({min(99, max(1, round(100 * p)))}%)"


def label_text(names: list[str] | dict[str, list[str]]) -> str:
    """Positive names joined; or, given categories, one line per category."""
    if isinstance(names, dict):
        return "\n".join(f"{c}: {', '.join(n) or 'None'}" for c, n in names.items())
    return ", ".join(names) if names else NO_POSITIVE


def render(
    version: dict,
    examples: list[tuple[str, list[str]]],
    labels: list[str] | None,
    section: str | None = None,
    absent: list[str] | None = None,
) -> list[dict]:
    """Chat messages for one target.

    ``examples``: (report text, positive label names) per retrieved study, best first.
    ``labels``: the target's positive label names (None for a version without labels).
    Either can instead be categories (a ``categories`` dict).
    ``section``: the section generated (a SECTION_NAMES key), for a version with
    {section}. ``absent``: the target's label names predicted absent, for a version
    with {absent} ("None" if empty). With {image}, the content is a list of parts:
    text, IMAGE_PART, text.
    """
    with_labels = uses_labels(version)
    assert with_labels == (labels is not None), "labels given iff the version uses them"
    assert uses_examples(version) == bool(examples), "examples given iff it shows them"
    assert uses_absent(version) == (absent is not None), "absent given iff it uses it"
    blocks = [
        version["example"].format(
            index=i,
            report=clean_report(report),
            **({"labels": label_text(names)} if with_labels else {}),
        )
        for i, (report, names) in enumerate(examples, start=1)
    ]
    fields = {"examples": "\n\n".join(blocks), "image": _IMAGE_MARK}
    if uses_section(version):
        if section not in SECTION_NAMES:
            raise SystemExit(f"no section name for {section!r} in this prompt version")
        fields["section"] = SECTION_NAMES[section]
    if with_labels:
        fields["labels"] = label_text(labels)
    if absent is not None:
        fields["absent"] = ", ".join(absent) or "None"
    text = version["user"].format(**fields).strip()
    if not uses_image(version):
        return [{"role": "user", "content": text}]
    before, after = text.split(_IMAGE_MARK)
    parts = [{"type": "text", "text": before}] if before else []
    parts.append(dict(IMAGE_PART))
    if after:
        parts.append({"type": "text", "text": after})
    return [{"role": "user", "content": parts}]
