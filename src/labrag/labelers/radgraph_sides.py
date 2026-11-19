"""Side (left, right, bilateral) of CheXbert-14 labels from RadGraph-XL report graphs.

A graph's entities carry tokens, a label (``Observation::definitely present``,
``Anatomy::definitely present``, ...) and relations (``modify``, ``located_at``,
``suggestive_of``) to other entity keys. An observation present or uncertain names a
label when its phrase (its tokens plus the observations modifying it) matches the
label's pattern; its side comes from the anatomy it is located at, the anatomy
modifying that or the observation, and the anatomy of observations suggestive of it
("bibasilar opacities, likely atelectasis": the side is on the opacities).
"""

import re

# Labels a side applies to -> pattern on the observation phrase (lowercase).
LATERAL_LABELS = {
    "Lung Opacity": r"opaci|infiltrat|airspace|densit|hazi",
    "Lung Lesion": r"nodul|mass|lesion",
    "Consolidation": r"consolidat",
    "Pneumonia": r"pneumonia",
    "Atelectasis": r"atelecta",
    "Pneumothorax": r"pneumothora",
    "Pleural Effusion": r"effusion",
    "Pleural Other": r"thicken|pleural scar",
    "Fracture": r"fractur",
}
PRESENT = ("Observation::definitely present", "Observation::uncertain")
SIDES = ("left", "right", "bilateral")


def normalize(text: str) -> str:
    """Report text without case, whitespace or punctuation: RadGraph-XL's text is
    re-tokenized ("line ." for "line."), so this is how a graph finds its report."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def phrase_side(text: str) -> set[str]:
    t = text.lower()
    found = set()
    if re.search(r"\bbilateral|\bbibasil|\bbiapical|\bboth\b", t):
        found.add("bilateral")
    if re.search(r"\bleft\b|lingul|retrocardiac", t):
        found.add("left")
    if re.search(r"\bright\b|middle lobe", t):
        found.add("right")
    return found


def label_sides(entities: dict) -> dict[str, str]:
    """label -> side over the report's present observations of that label; a label
    seen on both sides, or as bilateral, is bilateral. Labels without a side are left
    out."""

    def linked(key: str, relation: str) -> list[str]:
        return [k for k, e in entities.items() if [relation, key] in e["relations"]]

    def anatomy_of(key: str) -> list[str]:
        e = entities[key]
        anatomy = [k for r, k in e["relations"] if r == "located_at"]
        anatomy += [
            k
            for k in linked(key, "modify")
            if entities[k]["label"].startswith("Anatomy")
        ]
        return anatomy + [m for a in anatomy for m in linked(a, "modify")]

    found: dict[str, set[str]] = {}
    for key, e in entities.items():
        if e["label"] not in PRESENT:
            continue
        modifiers = [
            entities[k]["tokens"]
            for k in linked(key, "modify")
            if entities[k]["label"].startswith("Observation")
        ]
        phrase = " ".join([e["tokens"], *modifiers]).lower()
        names = [n for n, p in LATERAL_LABELS.items() if re.search(p, phrase)]
        if not names:
            continue
        anatomy = anatomy_of(key)
        for k in linked(key, "suggestive_of"):
            anatomy += anatomy_of(k)
        side = phrase_side(
            " ".join([phrase, *(entities[a]["tokens"] for a in anatomy)])
        )
        for n in names:
            found.setdefault(n, set()).update(side)
    out = {}
    for n, s in found.items():
        if "bilateral" in s or {"left", "right"} <= s:
            out[n] = "bilateral"
        elif s:
            out[n] = next(iter(s))
    return out


def sided(name: str, side: str | None) -> str:
    """A label as the prompt shows it: "Pleural Effusion (left)"."""
    return f"{name} ({side})" if side else name
