"""Consistency checks on the hand transcription of the PadChest location tree.

The count check pins every depth and count: in the paper each node's "study counts with
child labels" equals its own count plus its children's, so a mis-nested or mistyped row
breaks a sum. Names are checked against the label CSV. CUIs are checked only by eye.
"""

import ast
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytest

from labrag import paths

TABLE = Path(__file__).resolve().parents[1] / "src/labrag/data/padchest_locations.tsv"
PADCHEST_CSV = "padchest/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz"


@pytest.fixture(scope="module")
def tree():
    return pd.read_csv(
        TABLE, sep="\t", comment="#", dtype={"cui": str}, keep_default_na=False
    )


def test_ids_and_depths(tree):
    assert list(tree.id) == list(range(len(tree)))
    assert tree.parent_id.iloc[0] == "" and tree.depth.iloc[0] == 0
    depth = dict(zip(tree.id, tree.depth))
    for parent, d in zip(tree.parent_id.iloc[1:], tree.depth.iloc[1:]):
        assert depth[int(parent)] == d - 1


def test_counts_sum_over_children(tree):
    children = defaultdict(int)
    for parent, n in zip(tree.parent_id.iloc[1:], tree.studies_with_children.iloc[1:]):
        children[int(parent)] += int(n)
    for row in tree.iloc[1:].itertuples():
        assert int(row.studies) + children[row.id] == int(
            row.studies_with_children
        ), row.name


def test_repeated_names_agree(tree):
    # A location printed under two parents has the same CUI and own count both times
    # (its with-children count can differ, e.g. `upper lobe`).
    per_name = tree.iloc[1:].groupby("name")[["cui", "studies"]].nunique()
    assert (per_name == 1).all().all()


@pytest.mark.skipif(
    not (paths.CXR_ROOT / PADCHEST_CSV).exists(),
    reason="PadChest label CSV not mounted",
)
def test_names_match_label_csv(tree):
    locs = pd.read_csv(
        paths.CXR_ROOT / PADCHEST_CSV, usecols=["Localizations"]
    ).Localizations
    in_csv = {
        tag.strip().removeprefix("loc ")
        for value in locs.dropna()
        for tag in ast.literal_eval(re.sub(r"\bnan\b", "None", value))
        if tag
    }
    in_tree = set(tree.name.iloc[1:])
    assert in_csv - in_tree == {"lumbar vertebrae"}
    assert in_tree - in_csv == {"extracorporal"}
