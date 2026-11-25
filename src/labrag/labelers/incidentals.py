"""Incidental and chronic findings a report names, by rule: the labels stage's
``incidentals`` labeler.

Findings outside CheXbert-14 that reports state often (degenerative spine, aortic
calcification, low lung volumes, ...). Whitespace runs are collapsed first, so a
pattern's words match across a line wrap. A report is split into clauses
(severity.CLAUSE), and each finding a clause names gets a value: negative (0) if a
negation reaches it (scope.negated: "no hiatal hernia") or the clause then says it is
gone ("... appears normal"), uncertain (-1) if a hedge reaches it (scope.hedged:
"possible hiatal hernia") or it is one side of a differential ("atelectasis or
scarring"), else positive (1). A NORMAL phrase is a negative without naming the finding
("the lungs are well inflated"). Over a report: positive if any clause says so, else
uncertain, else negative, else blank (not mentioned).
"""

import re

import pandas as pd

from labrag.labelers import scope
from labrag.labelers.severity import CLAUSE

# Label -> pattern. Device-like findings (clips, sternotomy) are the devices labeler's.
INCIDENTALS = {
    "Degenerative Spine Changes": (
        r"degenerat|spondyl|osteophyt|\bdish\b|dis[ck] space"
        r"|interspaces? (\w+ ){0,3}narrow|intervertebral (dis[ck] )?space"
        r"|productive (\w+ )?changes? of the (\w+ )?spine"
    ),
    "Scoliosis or Kyphosis": r"scolio|kyphos|kyphotic",
    "Aortic Calcification": (
        r"atheroscl|(aort|arch|knob|vascular)\w*\s+(\w+\s+){0,4}calcif"
        r"|calcif\w*\s+(\w+\s+){0,4}(aort|arch|knob|vascul)"
    ),
    "Tortuous or Dilated Aorta": (
        r"tortu|unfold|\bectatic\b|\bectasia\b|aneurysm"
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
        r"low lung volume|low in volume"
        r"|lung volumes? ((are|is|have|has|remain|remains|appear|appears) )?(been )?"
        r"((?!not\b)\w+ )?(low|lower|reduced|decreased|diminished)"
        r"|low (inspiratory )?volumes|hypoinflat|low inspirat"
        r"|(poor|shallow|limited|suboptimal) inspir"
        r"|inspiration is (\w+ )?(poor|shallow|limited|suboptimal|low)"
    ),
    "Elevated Hemidiaphragm": (
        r"elevat\w* (of the )?(right |left )?hemidiaphragm"
        r"|hemidiaphragm (is |remains )?(\w+ )?elevat|eventration"
    ),
    "Chronic Interstitial Changes or Scarring": (
        r"fibros|fibrotic|\bscar|interstitial (lung )?(disease|abnormalit|change)"
        r"|chronic (\w+ ){0,2}interstitial|interstitial (\w+ ){0,10}chronic"
        r"|reticul|bronchiectas|honeycomb"
    ),
    "Osteopenia": r"osteopen|demineraliz|osteopor",
}
PATTERNS = {name: re.compile(p, re.I) for name, p in INCIDENTALS.items()}
NAMES = list(INCIDENTALS)

# Label -> a phrase that states the finding absent without naming it.
NORMAL = {
    "Low Lung Volumes": re.compile(
        r"(well|adequately|normally)[- ](inflated|expanded|aerated)"
        r"|normal lung volumes?|lung volumes? (are |is |appear |remain )?normal"
        r"|(good|adequate|normal) inspiration",
        re.I,
    ),
}
NOT_BEFORE = re.compile(r"\bnot\s+(\w+\s+)?$", re.I)
# "The right lung is well expanded" is a re-expanded lung, not lung volume.
ONE_SIDE = re.compile(r"\b(right|left)\b", re.I)
# The clause goes on to say the finding is gone.
GONE_AFTER = re.compile(
    r"\b(resolved|normalized|appears? normal"
    r"|no longer (seen|visible|evident|present))\b",
    re.I,
)
# Matches that name something else: subcutaneous emphysema, a disc hernia, an
# overinflated tube cuff, shoulder arthritis, a tortuous splenic artery.
NON_EMPHYSEMA = re.compile(
    r"(subcutaneous|sub-cutaneous|subq|mediastinal|soft[- ]tissues?|chest wall|surgical"
    r"|orbital|intraorbital|prevertebral|retropharyngeal|interstitial)"
    r"\W+(\w+\W+){0,2}$",
    re.I,
)
CUFF = re.compile(r"\bcuff", re.I)
NON_HIATAL = re.compile(
    r"(dis[ck]|uncal|tonsillar|inguinal|umbilical|ventral|incisional|bochdalek\W*s?"
    r"|morgagni|diaphragmatic|lung|parastomal|abdominal)\W+(\w+\W+)?$",
    re.I,
)
JOINT = re.compile(
    r"shoulder|acromioclavicular|\bac\b|glenohumeral|humer|sternoclavicular|clavic"
    r"|\bjoints?\b|\bhips?\b",
    re.I,
)
SPINE = re.compile(r"spin|vertebr|thoracic|lumbar|cervical|dis[ck]|interspace", re.I)
OTHER_VESSEL = re.compile(
    r"splenic|carotid|subclavian|innominate|brachiocephalic|iliac|ventric|arter(y|ies)",
    re.I,
)
AORTA = re.compile(r"aort|arch|knob", re.I)
DIFFERENTIAL = re.compile(r"\s*(?i:\bor\b|/|\bversus\b|\bvs\b\.?)\s*")


def names_other(name: str, clause: str, m: re.Match) -> bool:
    """m names something other than the label."""
    before = clause[: m.start()]
    if name == "Emphysema or Hyperinflation":
        if m.group(0).lower().startswith("emphysem"):
            return bool(NON_EMPHYSEMA.search(before))
        return bool(CUFF.search(clause))
    if name == "Hiatal Hernia":
        return bool(NON_HIATAL.search(before))
    if name == "Degenerative Spine Changes":
        return bool(JOINT.search(clause)) and not SPINE.search(clause)
    if name == "Tortuous or Dilated Aorta":
        return bool(OTHER_VESSEL.search(clause)) and not AORTA.search(clause)
    return False


def differential(pattern: re.Pattern, clause: str, m: re.Match) -> bool:
    """m is one side of "A or B" (or A/B, A versus B) whose other side is not the same
    finding: "atelectasis or scarring" (not "fibrosis or scarring")."""
    end = m.end() + re.match(r"\w*", clause[m.end() :]).end()
    left = re.search(r"(?P<w>\w+)" + DIFFERENTIAL.pattern + "$", clause[: m.start()])
    right = re.match(DIFFERENTIAL.pattern + r"(?P<w>\w+)", clause[end:])
    return any(s and not pattern.search(s["w"]) for s in (left, right))


def label(text: str) -> dict[str, float]:
    """One report's incidental labels: name -> 1.0 (positive), -1.0 (uncertain), 0.0
    (negative); names not mentioned are left out."""
    seen = {1.0: set(), -1.0: set(), 0.0: set()}
    for clause in CLAUSE.split(" ".join((text or "").split())):
        for name, pattern in PATTERNS.items():
            for m in pattern.finditer(clause):
                if names_other(name, clause, m):
                    continue
                gone = GONE_AFTER.search(scope.after(clause, m))
                if scope.negated(clause, m) or gone:
                    seen[0.0].add(name)
                elif (  # a hedge inside the match too: "interstitial ... may be chronic"
                    scope.hedged(clause, m)
                    or scope.HEDGE.search(m.group(0))
                    or differential(pattern, clause, m)
                ):
                    seen[-1.0].add(name)
                else:
                    seen[1.0].add(name)
        for name, pattern in NORMAL.items():
            if ONE_SIDE.search(clause):
                continue
            for m in pattern.finditer(clause):
                if not NOT_BEFORE.search(clause[: m.start()]):
                    seen[0.0].add(name)
    out, done = {}, set()
    for value, names in seen.items():  # positive, then uncertain, then negative
        out |= {n: value for n in names - done}
        done |= names
    return out


def label_reports(texts) -> pd.DataFrame:
    """Incidental labels of many reports: one row each, a column per NAMES (blank = not
    mentioned)."""
    return pd.DataFrame([label(t) for t in texts], columns=NAMES, dtype=float)
