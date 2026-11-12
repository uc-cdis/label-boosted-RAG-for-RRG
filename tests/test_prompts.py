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
