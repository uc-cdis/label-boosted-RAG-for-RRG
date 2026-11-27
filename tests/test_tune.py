"""tune stage: recipes and inputs (no GPU, no Docker)."""

import pytest

from labrag import serving
from labrag.stages import tune


@pytest.mark.parametrize("name", sorted(tune.load_config()))
def test_recipe_names_known_entries(name):
    recipe = tune.load_config()[name]
    assert "vision" in serving.load_config("models")[recipe["model"]]
    assert recipe["sampling"] in serving.load_config("sampling")
    assert set(recipe["reward_weights"]) == {"ratescore", "cxrbert", "bertscore", "arn"}
    grpo = recipe["grpo"]
    assert grpo["max_completion_length"] < grpo["vllm_max_model_length"]


def test_prompts_must_be_one_train_artifact_per_dataset():
    a = "train/mimic/findings+impression/labrag.x/joint-simple"
    with pytest.raises(SystemExit, match="one prompt artifact per dataset"):
        tune.run([a, a], "grpo-thinking-smoke")
    with pytest.raises(SystemExit, match="prompts/train"):
        tune.run(
            ["val/mimic/findings+impression/labrag.x/joint-simple"],
            "grpo-thinking-smoke",
        )
