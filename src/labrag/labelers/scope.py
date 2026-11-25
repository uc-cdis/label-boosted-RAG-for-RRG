"""How far a negation or hedge reaches in a clause: shared by the devices and
incidentals rule labelers.

A trigger (a negation: severity.NEGATED, "no hiatal hernia"; a hedge: HEDGE,
"possible hiatal hernia") reaches forward through its clause until a SCOPE_END: a turn
("no fracture, noting degenerative changes"), a location ("no free air below the
elevated hemidiaphragm"), a "no change in" phrase (the finding persists), or a
capitalized word (a line break lost to whitespace collapsing: "No pneumothorax There is
..."). A hedge after the finding counts too ("hernia is suspected", "fibrosis is less
likely").
"""

import re

from labrag.labelers.severity import NEGATED

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
HEDGE = re.compile(
    r"\b(?:possible|possibly|probable|probably|likely|may|might|could|suspect|suspected"
    r"|suspicious for|suspicion (?:for|of)|concern(?:ing)? for|question of|questionable"
    r"|questioned|cannot (?:be )?exclude|can ?not (?:be )?exclude|can't (?:be )?exclude"
    r"|presumed|presumably|suggest|suggests|suggested|suggesting|suggestive"
    r"|indeterminate|equivocal|versus|vs|differential)\b",
    re.I,
)
HEDGE_AFTER = re.compile(
    r"\b(?:(?:is|are) (?:suspected|questioned|questionable|possible|less likely"
    r"|unlikely)"
    r"|(?:cannot|can ?not|can't) be (?:excluded|ruled out)"
    r"|not (?:be )?(?:excluded|ruled out))\b",
    re.I,
)
# "High-resolution CT" negates nothing.
HIGH_RESOLUTION = re.compile(r"high[- ]?$", re.I)


def reaches(trigger: re.Pattern, clause: str, m: re.Match) -> bool:
    """trigger's last match before m reaches it: no SCOPE_END between, and m doesn't
    start a new line itself ("No pneumothorax Vascular calcification")."""
    hits = [
        t
        for t in trigger.finditer(clause[: m.start()])
        if not HIGH_RESOLUTION.search(clause[: t.start()])
    ]
    if not hits:
        return False
    scope = clause[hits[-1].end() : m.start()]
    new_line = scope.strip() and re.match(r"[A-Z][a-z]", m.group(0))
    return not (SCOPE_END.search(scope) or new_line)


def after(clause: str, m: re.Match) -> str:
    """The clause after m, up to the next SCOPE_END."""
    rest = clause[m.end() :]
    end = SCOPE_END.search(rest)
    return rest[: end.start()] if end else rest


def negated(clause: str, m: re.Match) -> bool:
    return reaches(NEGATED, clause, m)


def hedged(clause: str, m: re.Match) -> bool:
    return reaches(HEDGE, clause, m) or bool(HEDGE_AFTER.search(after(clause, m)))
