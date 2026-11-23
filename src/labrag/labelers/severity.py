"""The size or severity a report gives a positive label, by rule: the severity stage.

A report is split into clauses (CLAUSE). A clause speaks of a label if it has one of the
label's KEYWORDS and no negation before it ("no large effusion" says nothing about an
effusion's size). Its severity is the SEVERITY phrase in that clause nearest to the
keyword ("small", "mild to moderate"; adverbs as adjectives: "mildly enlarged" is
"mild"). A label gets its distinct phrases in report order, at most MAX_PHRASES.
"""

import re

CLAUSE = re.compile(r"[.;:,]|\band\b|\bbut\b|\bwith\b|\bwhich\b|\bwhile\b", re.I)
NEGATED = re.compile(r"\b(no|without|resolved|resolution|negative for|free of)\b", re.I)
_WORD = (
    r"trace|tiny|minimal(?:ly)?|small|mild(?:ly)?|moderate(?:ly)?|large|severe(?:ly)?"
    r"|massive|extensive|marked(?:ly)?|borderline"
)
SEVERITY = re.compile(
    rf"\b({_WORD})(?:(?:\s*-?\s*\b(?:to|or)\b\s*-?\s*|\s*-\s*)({_WORD}))?(?:-sized?)?\b(?![-\w])",
    re.I,
)
ADJECTIVE = {
    "minimally": "minimal",
    "mildly": "mild",
    "moderately": "moderate",
    "severely": "severe",
    "markedly": "marked",
}
# CheXbert-14 labels whose size or severity reports state, and the words for them.
KEYWORDS = {
    "Pleural Effusion": r"effusion|pleural fluid",
    "Pneumothorax": r"pneumothora",
    "Cardiomegaly": (
        r"cardiomegaly|heart size|cardiac silhouette|cardiac enlargement"
        r"|heart is enlarged|enlarged heart|enlargement of the heart"
    ),
    "Edema": r"edema|congestion",
    "Atelectasis": r"atelecta|collapse",
    "Consolidation": r"consolidat",
    "Lung Opacity": r"opacit|densit",
    "Pneumonia": r"pneumonia",
    "Enlarged Cardiomediastinum": r"mediastin",
}
PATTERNS = {name: re.compile(p, re.I) for name, p in KEYWORDS.items()}
MAX_PHRASES = 2


def _phrase(m: re.Match) -> str:
    words = [ADJECTIVE.get(w.lower(), w.lower()) for w in m.groups() if w]
    return " to ".join(dict.fromkeys(words))


def severities(text: str, names) -> dict[str, str]:
    """Each of ``names`` (labels, KEYWORDS ones only) -> its severity text in report
    ``text`` ("small", "mild to moderate", "small, moderate"); labels without one
    are left out."""
    found = {n: [] for n in names if n in PATTERNS}
    for clause in CLAUSE.split(text or ""):
        phrases = list(SEVERITY.finditer(clause))
        if not phrases:
            continue
        for name in found:
            for k in PATTERNS[name].finditer(clause):
                if NEGATED.search(clause[: k.start()]):
                    continue
                near = min(
                    phrases,
                    key=lambda m: min(
                        abs(m.start() - k.end()), abs(k.start() - m.end())
                    ),
                )
                found[name].append(_phrase(near))
                break
    return {
        n: ", ".join(list(dict.fromkeys(p))[:MAX_PHRASES])
        for n, p in found.items()
        if p
    }
