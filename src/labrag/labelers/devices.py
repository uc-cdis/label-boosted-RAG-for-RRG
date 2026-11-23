"""Device types a report names, by rule: the labels stage's ``devices`` labeler.

A report is split into clauses (CLAUSE). In each clause the DEVICES patterns are tried
in order, and each match is cut out before the next pattern is tried, so a PICC is not
also a central line. A device is positive (1) if some clause names it without saying it
is gone, negative (0) if every clause that names it says so (REMOVED, or "no" right
before it: "no endotracheal tube"), else blank (not mentioned).
"""

import re

import pandas as pd

# Clause boundaries: a device and a removal word in different clauses don't meet.
CLAUSE = re.compile(r"[.;:,]|\band\b|\bbut\b|\bwhile\b|\bwhereas\b", re.I)
REMOVED = re.compile(
    r"remov|withdrawn|no longer|discontinu|extubat|decannulat|pulled|taken out", re.I
)
NO_BEFORE = re.compile(r"\b(no|without)\s+(\w+\s+)?$", re.I)

# Label -> pattern, in matching order (more specific first).
DEVICES = {
    "PICC Line": r"\bpicc\b|peripherally inserted",
    "Pulmonary Artery Catheter": r"swan|pulmonary arter(y|ial) catheter",
    "Chest Port": r"port-?a-?cath|porta-?cath|mediport|power ?port|\bport\b",
    "Central Venous Catheter": (
        r"central (venous )?(line|catheter|access)|central venous|\bcvc\b|\bcvl\b"
        r"|\bij\b|internal jugular|subclavian (\w+ )?(line|catheter|approach)"
        r"|femoral (\w+ )?(line|catheter)|(hemo)?dialysis (\w+ )?(line|catheter)"
        r"|tunnel+ed (\w+ )?(line|catheter)|introducer|cordis|(triple|double)[- ]lumen"
    ),
    "Endotracheal Tube": r"endotracheal|\bett\b|\bet\b(?=[\s-]*(tube|tip))|\be\.t\.",
    "Tracheostomy Tube": r"tracheostomy|\btrach\b",
    "Enteric Tube": (
        r"nasogastric|orogastric|nasoenteric|naso-?jejunal|nasoduodenal"
        r"|\b(ng|og|ngt|ogt)\b|enteric|feeding tube|dobb?hoff|gastric tube"
        r"|post-?pyloric|gastrostomy|\bg-?tube\b|\bpeg\b"
    ),
    "Chest Tube": (
        r"chest tube|thoracostomy|pleural (drain|catheter|tube)|pigtail|pleurx"
        r"|chest drain"
    ),
    "Cardiac Valve Device": (
        r"valve (replacement|repair|prosthes)|(prosthetic|mechanical|bioprosthetic) "
        r"(\w+ )?valve|\b(tavr|tavi|avr|mvr)\b|annuloplasty|mitra-?clip|mitral (ring|clip)"
        r"|watchman"
    ),
    "Pacemaker or Defibrillator": (
        r"pacemaker|pacer|\ba?icd\b|defibrillator|pacing|biventricular|\bcrt\b"
        r"|loop recorder"
    ),
    "Circulatory Support Device": (
        r"\blvad\b|ventricular assist|impella|\becmo\b|intra-?aortic balloon|\biabp\b"
        r"|balloon pump"
    ),
    "Sternotomy Wires": r"sternotomy|sternal (wires?|sutures?|cerclage)",
    "Surgical Clips": r"\bclips?\b|\bstaples?\b",
    "Stent": r"\bstents?\b",
    "Monitoring Leads": (
        r"(ekg|ecg|monitoring|telemetry|cardiac monitor\w*) (leads?|wires?|electrodes?)"
        r"|\bleads? overl"
    ),
}
PATTERNS = {name: re.compile(p, re.I) for name, p in DEVICES.items()}
NAMES = list(DEVICES)


def label(text: str) -> dict[str, float]:
    """One report's device labels: name -> 1.0 (named), 0.0 (named as gone); names
    not mentioned are left out."""
    named, gone = set(), set()
    for clause in CLAUSE.split(text or ""):
        removed = bool(REMOVED.search(clause))
        for name, pattern in PATTERNS.items():
            hits = list(pattern.finditer(clause))
            for m in hits:
                if removed or NO_BEFORE.search(clause[: m.start()]):
                    gone.add(name)
                else:
                    named.add(name)
            if hits:  # cut the matches out: later patterns don't see them
                clause = pattern.sub(" ", clause)
    return {n: 1.0 for n in named} | {n: 0.0 for n in gone - named}


def label_reports(texts) -> pd.DataFrame:
    """Device labels of many reports: one row each, a column per NAMES (blank = not
    mentioned)."""
    return pd.DataFrame([label(t) for t in texts], columns=NAMES, dtype=float)
