import pytest

from labrag.labelers.radgraph_sides import label_sides, normalize, sided

P, U, A = (
    "Observation::definitely present",
    "Observation::uncertain",
    "Anatomy::definitely present",
)
NEG = "Observation::definitely absent"


def ent(tokens, label, *relations):
    return {"tokens": tokens, "label": label, "relations": [list(r) for r in relations]}


def test_effusion_located_at_left_base():
    g = {
        "1": ent("left", A, ("modify", "2")),
        "2": ent("base", A),
        "3": ent("effusion", P, ("located_at", "2")),
    }
    assert label_sides(g) == {"Pleural Effusion": "left"}


def test_side_from_suggestive_opacity():
    # "bibasilar opacities, likely atelectasis"
    g = {
        "1": ent("bibasilar", A),
        "2": ent("opacities", P, ("located_at", "1"), ("suggestive_of", "3")),
        "3": ent("atelectasis", U),
    }
    assert label_sides(g) == {"Lung Opacity": "bilateral", "Atelectasis": "bilateral"}


def test_both_sides_are_bilateral_and_absent_is_ignored():
    g = {
        "1": ent("left", A),
        "2": ent("effusion", P, ("located_at", "1")),
        "3": ent("right", A),
        "4": ent("effusion", P, ("located_at", "3")),
        "5": ent("pneumothorax", NEG, ("located_at", "3")),
    }
    assert label_sides(g) == {"Pleural Effusion": "bilateral"}


@pytest.mark.parametrize(
    "anatomy, side",
    [
        ("retrocardiac", "left"),
        ("lingula", "left"),
        ("right middle lobe", "right"),
        ("lung", None),
    ],
)
def test_anatomy_words(anatomy, side):
    g = {"1": ent(anatomy, A), "2": ent("consolidation", P, ("located_at", "1"))}
    assert label_sides(g).get("Consolidation") == side


def test_text_matching_ignores_tokenization():
    assert normalize("Left PICC line. No effusion.") == normalize(
        "Left PICC line . No effusion ."
    )


def test_sided_label_text():
    assert sided("Pleural Effusion", "left") == "Pleural Effusion (left)"
    assert sided("Cardiomegaly", None) == "Cardiomegaly"
