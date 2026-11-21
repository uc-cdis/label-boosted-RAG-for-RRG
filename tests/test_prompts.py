"""Prompt versions and rendering."""

import pytest

from labrag import prompts


@pytest.mark.parametrize("name", ["preprint-naive", "preprint-simple"])
def test_preprint_versions_load(name):
    version = prompts.load(name)
    assert prompts.uses_labels(version) == (name == "preprint-simple")
    assert "<<Report>>" not in version["user"]


def test_render_simple():
    version = prompts.load("preprint-simple")
    examples = [("Heart is  \nnormal.\n\nNo  effusion.", ["No Finding"]), ("x", [])]
    [message] = prompts.render(version, examples, ["Cardiomegaly", "Edema"])
    assert message["role"] == "user"
    text = message["content"]
    assert (
        "CONTEXT:\nExample: 1\nLabel: No Finding\nHeart is normal. No effusion.\n\n"
        "Example: 2\nLabel: Other\nx\n\nNow generate" in text
    )
    assert text.endswith("examples shown.\nLabel: Cardiomegaly, Edema")


def test_render_naive_has_no_labels():
    version = prompts.load("preprint-naive")
    [message] = prompts.render(version, [("a {b} c", [])], None)
    assert "CONTEXT:\nExample: 1\na {b} c\n\nNow generate" in message["content"]
    assert "Label" not in message["content"]
    with pytest.raises(AssertionError):
        prompts.render(version, [("a", [])], ["Edema"])


def test_render_indication_for_the_target_only():
    version = prompts.load("section-simple-negatives-indication-image")
    assert prompts.uses_indication(version)
    [message] = prompts.render(
        version,
        [("a", ["Edema"])],
        ["Edema"],
        "findings",
        indication=" Fall,\n rib pain ",
    )
    first, _, _ = message["content"]
    assert first["text"].endswith("CURRENT CASE:\nIndication: Fall, rib pain\n")
    assert "Indication" not in first["text"].split("CONTEXT:")[1].split("CURRENT")[0]
    [message] = prompts.render(version, [("a", [])], [], "findings", indication="")
    assert message["content"][0]["text"].endswith("Indication: Not provided\n")
    with pytest.raises(AssertionError):
        prompts.render(version, [("a", [])], [], "findings")
    plain = prompts.load("section-simple-negatives-image")
    with pytest.raises(AssertionError):
        prompts.render(plain, [("a", [])], [], "findings", indication="x")


def test_every_version_loads():
    for name in prompts.yaml.safe_load(prompts.PROMPTS_YAML.read_text()):
        prompts.load(name)


def test_render_image_parts():
    version = prompts.load("section-simple-image")
    [message] = prompts.render(version, [("a", ["Edema"])], ["Edema"], "findings")
    before, image, after = message["content"]
    assert image == prompts.IMAGE_PART
    assert before["text"].endswith("Label: Edema\na\n\nCURRENT CASE:\n")
    assert after["text"].startswith("\n\nNow generate") and "\x00" not in after["text"]
    assert prompts.text_chars([message]) == len(before["text"]) + len(after["text"])


def test_render_image_only_takes_no_examples():
    version = prompts.load("section-image")
    assert not prompts.uses_examples(version) and "example" not in version
    [message] = prompts.render(version, [], None, "findings")
    assert message["content"][1] == prompts.IMAGE_PART
    with pytest.raises(AssertionError):
        prompts.render(version, [("a", [])], None, "findings")


@pytest.mark.parametrize("section", ["findings", "impression"])
def test_render_names_the_section(section):
    version = prompts.load("section-naive")
    [message] = prompts.render(version, [("a", [])], None, section)
    name = prompts.SECTION_NAMES[section]
    assert f"5. Write only the {name} section of the report" in message["content"]
    with pytest.raises(SystemExit):
        prompts.render(version, [("a", [])], None, "findings_impression")


def test_probability_label_text_is_clamped():
    assert prompts.with_probability("Atelectasis", 0.284) == "Atelectasis (28%)"
    assert prompts.with_probability("Support Devices", 0.999) == "Support Devices (99%)"
    assert prompts.with_probability("Pneumonia", 0.001) == "Pneumonia (1%)"


def test_categories_render_one_line_each():
    names = ["Edema", "Pneumothorax", "Atelectasis", "No Finding"]
    codes = [1.0, 0.0, float("nan"), None]
    every = ["Positive", "Negative", "Uncertain", "Unmentioned"]
    assert prompts.label_text(prompts.categories(codes, names, every)) == (
        "Positive: Edema\nNegative: Pneumothorax\nUncertain: None\n"
        "Unmentioned: Atelectasis, No Finding"
    )
    some = prompts.categories(codes, names, ["Positive", "Uncertain"])
    assert prompts.label_text(some) == "Positive: Edema\nUncertain: None"
    for name, shown in [
        ("section-pu-image", ["Positive", "Uncertain"]),
        ("section-pnu-image", ["Positive", "Negative", "Uncertain"]),
        ("section-verbose-image", every),
        ("section-instruct-image", every),
    ]:
        assert prompts.load(name)["label_categories"] == shown
    assert not prompts.uses_categories(prompts.load("section-simple-image"))


def test_render_absent_lists_predicted_negatives():
    version = prompts.load("section-simple-absent-image")
    assert prompts.uses_absent(version) and not prompts.uses_categories(version)
    [message] = prompts.render(
        version, [("a", ["Edema"])], ["Edema"], "findings", ["Pneumothorax", "Fracture"]
    )
    assert message["content"][-1]["text"].endswith(
        "Label: Edema\nAbsent: Pneumothorax, Fracture"
    )
    [message] = prompts.render(version, [("a", [])], [], "findings", [])
    assert message["content"][-1]["text"].endswith("Label: Other\nAbsent: None")
    with pytest.raises(AssertionError):
        prompts.render(version, [("a", [])], [], "findings")
    plain = prompts.load("section-simple-negatives-image")
    with pytest.raises(AssertionError):
        prompts.render(plain, [("a", [])], [], "findings", [])
