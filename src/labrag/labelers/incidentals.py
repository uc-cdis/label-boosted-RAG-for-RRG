"""Incidental and chronic findings a report names, by rule: the labels stage's
``incidentals`` labeler.

Findings outside CheXbert-14 that reports state often (degenerative spine, aortic
calcification, low lung volumes, ...). Whitespace runs are collapsed first, so a
pattern's words match across a line wrap. A report is split into clauses
(severity.CLAUSE).
A finding is positive (1) if some clause names it without a negation reaching it,
negative (0) if every clause that names it negates it, else blank (not mentioned). A
negation (severity.NEGATED: "no hiatal hernia") reaches forward through its clause until
a SCOPE_END: a turn ("no fracture, noting degenerative changes"), a location ("no free
air below the elevated hemidiaphragm"), a "no change in" phrase (the finding persists),
or a capitalized word (a lost line break: "No pneumothorax There is ...").
"""

import re

import pandas as pd

from labrag.labelers.severity import CLAUSE, NEGATED

# Label -> pattern. Device-like findings (clips, sternotomy) are the devices labeler's.
INCIDENTALS = {
    "Degenerative Spine Changes": r"degenerat|spondyl|osteophyt|\bdish\b|disc space",
    "Scoliosis or Kyphosis": r"scolio|kyphos|kyphotic",
    "Aortic Calcification": (
        r"atheroscl|(aort|arch|knob|vascular)\w*\s+(\w+\s+){0,3}calcif"
        r"|calcif\w*\s+(\w+\s+){0,3}(aort|arch|knob|vascul)"
    ),
    "Tortuous or Dilated Aorta": (
        r"tortu|unfold|ectatic|ectasia|aneurysm"
        r"|aort\w*\s+(\w+\s+){0,2}(dilat|enlarg|widen)"
        r"|(dilat|enlarg|widen)\w*\s+(\w+\s+){0,2}aort"
    ),
    "Hiatal Hernia": r"hernia",
    "Emphysema or Hyperinflation": (
        r"emphysem|hyperinfla|hyperexpan|copd|chronic obstructive|overinfla"
        r"|flatten\w* (of the )?(\w+ )?(hemi)?diaphragm"
    ),
    "Calcified Granuloma": (
        r"granulom|calcified (pulmonary )?(nodul|lymph|node|focus|foci|densit)"
    ),
    "Low Lung Volumes": (
        r"low lung volume|lung volumes are (\w+ )?(low|reduced|decreased|diminished)"
        r"|low (inspiratory )?volumes|hypoinflat|poor inspirat|shallow inspir"
        r"|low inspirat"
    ),
    "Elevated Hemidiaphragm": (
        r"elevat\w* (of the )?(right |left )?hemidiaphragm"
        r"|hemidiaphragm (is |remains )?(\w+ )?elevat|eventration"
    ),
    "Chronic Interstitial Changes or Scarring": (
        r"fibros|fibrotic|\bscar|interstitial (lung )?(disease|abnormalit|marking|change)"
        r"|reticul|bronchiectas|honeycomb"
    ),
    "Osteopenia": r"osteopen|demineraliz|osteopor",
}
PATTERNS = {name: re.compile(p, re.I) for name, p in INCIDENTALS.items()}
SCOPE_END = re.compile(
    r"(?i:\b(?:though|although|however|except|other than|aside from|apart from"
    r"|short of|noting|given|setting of|allowing for|accounting for|accounted for"
    r"|attributable to|due to|because of|secondary to|result of|related to|limited by"
    r"|limitation|based on|in this|for this|despite|superimposed (?:on|upon)"
    r"|background of|including|below|beneath|under|underneath|above|behind|overlying"
    r"|obscured by|there (?:is|are|was|were))\b"
    r"|\b(?:change|increase|progression|improvement|worsening)\s+(?:in|of|to|since)\b)"
    r"|\b[A-Z][a-z]+\b|\d\)"
)
NAMES = list(INCIDENTALS)


def label(text: str) -> dict[str, float]:
    """One report's incidental labels: name -> 1.0 (named), 0.0 (named, negated);
    names not mentioned are left out."""
    named, negated = set(), set()
    for clause in CLAUSE.split(" ".join((text or "").split())):
        for name, pattern in PATTERNS.items():
            for m in pattern.finditer(clause):
                if negated_at(clause, m):
                    negated.add(name)
                else:
                    named.add(name)
    return {n: 1.0 for n in named} | {n: 0.0 for n in negated - named}


def negated_at(clause: str, m: re.Match) -> bool:
    """A negation earlier in the clause reaches match m: no SCOPE_END between, and m
    doesn't start a new line itself ("No pneumothorax Vascular calcification")."""
    triggers = list(NEGATED.finditer(clause[: m.start()]))
    if not triggers:
        return False
    scope = clause[triggers[-1].end() : m.start()]
    new_line = scope.strip() and re.match(r"[A-Z][a-z]", m.group(0))
    return not (SCOPE_END.search(scope) or new_line)


def label_reports(texts) -> pd.DataFrame:
    """Incidental labels of many reports: one row each, a column per NAMES (blank = not
    mentioned)."""
    return pd.DataFrame([label(t) for t in texts], columns=NAMES, dtype=float)
