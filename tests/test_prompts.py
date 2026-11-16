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
