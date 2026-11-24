"""Incidental and chronic findings a report names, by rule: the labels stage's
``incidentals`` labeler.

Findings outside CheXbert-14 that reports state often (degenerative spine, aortic
calcification, low lung volumes, ...). A report is split into clauses (severity.CLAUSE).
A finding is positive (1) if some clause names it without a negation before it in the
clause (severity.NEGATED: "no hiatal hernia"), negative (0) if every clause that names
it negates it, else blank (not mentioned).
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
NAMES = list(INCIDENTALS)


def label(text: str) -> dict[str, float]:
    """One report's incidental labels: name -> 1.0 (named), 0.0 (named, negated);
    names not mentioned are left out."""
    named, negated = set(), set()
    for clause in CLAUSE.split(text or ""):
        for name, pattern in PATTERNS.items():
            for m in pattern.finditer(clause):
                if NEGATED.search(clause[: m.start()]):
                    negated.add(name)
                else:
                    named.add(name)
    return {n: 1.0 for n in named} | {n: 0.0 for n in negated - named}


def label_reports(texts) -> pd.DataFrame:
    """Incidental labels of many reports: one row each, a column per NAMES (blank = not
    mentioned)."""
    return pd.DataFrame([label(t) for t in texts], columns=NAMES, dtype=float)
