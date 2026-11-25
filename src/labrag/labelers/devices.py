"""Device types a report names, by rule: the labels stage's ``devices`` labeler.

Whitespace runs are collapsed first, so a pattern's words match across a line wrap
(MIMIC-CXR and CheXpert Plus reports are wrapped). A report is split into clauses
(CLAUSE). In each clause the DEVICES patterns are tried in order, and each match is cut
out before the next pattern is tried, so a PICC is not also a central line.

In a clause, a device is negative (0) if the clause says it is gone (REMOVED, judged on
the full text so "withdrawn and now lies ..." stays a repositioning) or a negation is
right before it (NO_BEFORE: "no endotracheal tube", "no ET or enteric tube"),
uncertain (-1) if a hedge within a few words reaches it ("possible PICC"; not "I
suspect the ET tube tip lies ...", which hedges where it is), else positive (1).
GONE_ALONE words say a device is gone without naming it ("extubated"). Over a report:
positive if any clause says so, else uncertain, else negative, else blank (not
mentioned).
"""

import re

import pandas as pd

from labrag.labelers import scope

# Clause boundaries: a device and a removal word in different clauses don't meet.
CLAUSE = re.compile(r"[.;:,]|\band\b|\bbut\b|\bwhile\b|\bwhereas\b", re.I)
# "Pulled back", "partially withdrawn", "should be withdrawn 2 cm" reposition a device.
REMOVED = re.compile(
    r"remov|no longer|discontinu|extubat|decannulat|taken out|pulled out"
    r"|(?<!be )(?<!partially )(?<!slightly )(?<!minimally )(?<!somewhat )withdrawn"
    r"(?!\s+(back|slightly|partially|minimally|somewhat|further|by|approximately|about"
    r"|\d|a few|so|to|and|into|proximally))",
    re.I,
)
# "no" + at most one word, through "or" lists; "no change in the ETT" negates nothing.
NO_BEFORE = re.compile(
    r"\b(no|without)\s+(?!(\w+\s+){0,2}change)((\w+\s+){1,3}(or|nor)\s+)*(\w+\s+)?$"
    r"|\bnot\s+(\w+\s+){0,3}$",
    re.I,
)
GONE_ALONE = {
    "Endotracheal Tube": re.compile(r"extubat", re.I),
    "Tracheostomy Tube": re.compile(r"decannulat", re.I),
}
# After a device: where it is. A hedge then is on the position, not the device.
POSITION = re.compile(
    r"\b(tip|terminat|lies|ends|position|projects|courses|extends)", re.I
)
# A hedge further back is on something else ("probable edema or ARDS NG tube").
HEDGE_WORDS = 3
# A pulmonary artery catheter's access site is not another central line.
PA_ACCESS = re.compile(
    r"\b((right|left)\s+)?(ij|internal jugular|jugular|subclavian|femoral)"
    r"(\s+(vein|venous))?(\s+(approach|access))?\b",
    re.I,
)

# Label -> pattern, in matching order (more specific first).
DEVICES = {
    "PICC Line": r"\bpicc\b|peripherally inserted",
    "Pulmonary Artery Catheter": r"swan|pulmonary arter(y|ial) catheter",
    "Chest Port": (  # not an enteric tube's side port
        r"port-?a-?cath|porta-?cath|mediport|power ?port"
        r"|(?<!side )(?<!side-)(?<!distal )(?<!proximal )\bport\b"
    ),
    "Central Venous Catheter": (
        r"central (venous )?(line|catheter|access)|central venous|\bcvc\b|\bcvl\b"
        r"|\bij\b|internal jugular|jugular (\w+ )?(line|catheter|approach)"
        r"|subclavian (\w+ )?(line|catheter|approach)|\buvc\b|umbilical venous"
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
        r"(\w+ )?valve|\b(tavr|tavi|avr|mvr)\b|annuloplasty|valvuloplasty ring"
        r"|mitra-?clip|mitral (ring|clip)"
        r"|watchman"
    ),
    "Pacemaker or Defibrillator": (
        r"pacemaker|(pacer|pacing|defibrillator)(?!\s*pads?)|\ba?icd\b|biventricular"
        r"|\bcrt\b"
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


def clause_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of each clause of text."""
    spans, start = [], 0
    for m in CLAUSE.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    return [*spans, (start, len(text))]


def hedged(clause: str, m: re.Match) -> bool:
    """A hedge reaches device match m: right before it, or after it ("is suspected"),
    unless the clause goes on to place the device."""
    if POSITION.search(clause[m.end() :]):
        return False
    hits = list(scope.HEDGE.finditer(clause[: m.start()]))
    near = hits and len(clause[hits[-1].end() : m.start()].split()) <= HEDGE_WORDS
    return bool(near and scope.reaches(scope.HEDGE, clause, m)) or bool(
        scope.HEDGE_AFTER.search(scope.after(clause, m))
    )


def label(text: str) -> dict[str, float]:
    """One report's device labels: name -> 1.0 (positive), -1.0 (uncertain), 0.0
    (negative); names not mentioned are left out."""
    text = " ".join((text or "").split())
    removals = [m.start() for m in REMOVED.finditer(text)]
    seen = {1.0: set(), -1.0: set(), 0.0: set()}
    for start, end in clause_spans(text):
        clause = text[start:end]
        removed = any(start <= r < end for r in removals)
        for name, pattern in GONE_ALONE.items():
            if pattern.search(clause):
                seen[0.0].add(name)
        for name, pattern in PATTERNS.items():
            hits = list(pattern.finditer(clause))
            for m in hits:
                if removed or NO_BEFORE.search(clause[: m.start()]):
                    seen[0.0].add(name)
                elif hedged(clause, m):
                    seen[-1.0].add(name)
                else:
                    seen[1.0].add(name)
            if hits:  # cut the matches out: later patterns don't see them
                clause = pattern.sub(" ", clause)
                if name == "Pulmonary Artery Catheter":
                    clause = PA_ACCESS.sub(" ", clause)
    out, done = {}, set()
    for value, names in seen.items():  # positive, then uncertain, then negative
        out |= {n: value for n in names - done}
        done |= names
    return out


def label_reports(texts) -> pd.DataFrame:
    """Device labels of many reports: one row each, a column per NAMES (blank = not
    mentioned)."""
    return pd.DataFrame([label(t) for t in texts], columns=NAMES, dtype=float)
