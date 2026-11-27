"""Generated reports as the metrics score them: whitespace collapsed, section headers
removed. Standard library only: the GRPO trainer image copies this file, so its reward
scores exactly the text evaluate scores."""

import re

JOINT_PARTS = ("findings", "impression")  # data.JOINT_PARTS (a test checks)
FINDINGS_HEADER = re.compile(r"^findings\s*:\s*", re.IGNORECASE)
# A JOINT header, optionally in markdown bold: "Impression:", "**Impression:**", ...
_HEADER = r"(?:\*\*)?{}(?:\*\*)?\s*:(?:\*\*)?\s*"
JOINT_HEADERS = [
    re.compile(_HEADER.format(name), re.IGNORECASE) for name in JOINT_PARTS
]
# Every metric scores an empty report as this text (the preprint did too, as "EMTPY").
EMPTY_REPORT = "EMPTY"


def format_report(text: str) -> str:
    """Findings text as scored: whitespace collapsed, a leading "Findings:" removed."""
    return FINDINGS_HEADER.sub("", " ".join(text.split()), count=1)


def split_joint(text: str) -> tuple[str, str] | None:
    """(findings, impression) of a generated Findings+Impression report, whitespace collapsed; None
    unless exactly one Findings header starts it, exactly one Impression header follows
    it, and neither section is empty."""
    text = " ".join(text.split())
    found = [list(h.finditer(text)) for h in JOINT_HEADERS]
    if [len(f) for f in found] != [1, 1]:
        return None
    (f,), (i,) = found
    if f.start() != 0 or i.start() < f.end():
        return None
    parts = text[f.end() : i.start()].strip(), text[i.end() :].strip()
    return parts if all(parts) else None


def format_joint(text: str) -> tuple[str, bool]:
    """A generated Findings+Impression report as scored, and whether the strict parser failed."""
    parts = split_joint(text)
    if parts:
        return " ".join(parts), False
    for h in JOINT_HEADERS:
        text = h.sub(" ", text)
    return " ".join(text.split()), True
