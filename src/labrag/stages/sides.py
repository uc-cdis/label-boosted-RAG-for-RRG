"""Stage ``sides``: the side (left, right, bilateral) of each study's mentioned labels.

Writes ``sides/<dataset>/<section>/<labeler>/sides.csv``: study_id, label, side, one row
per label the labeler mentions (positive or uncertain) whose side the report's RadGraph-XL
graph gives (labelers/radgraph_sides.py). The graphs are the dataset authors' release
(CheXpert Plus only); a graph finds its study by normalized report text.
"""

import json
import zipfile
from pathlib import Path

import pandas as pd

from labrag import artifacts, paths
from labrag.data import DATASETS, check_section
from labrag.labelers import radgraph_sides

LABELER = "chexbert14"
# dataset -> its RadGraph-XL release (a zip of section_<section>.json), under cxr_root
RELEASES = {"chexpertplus": "chexpertplus/radgraph-XL-annotations.zip"}


def run(
    dataset: str,
    section: str,
    overwrite: bool = False,
    cxr_root: Path | None = None,
) -> Path:
    check_section("sides", dataset, section)
    if dataset not in RELEASES:
        raise SystemExit(f"sides: no RadGraph-XL release for {dataset}")
    cxr_root = cxr_root or paths.CXR_ROOT
    adapter = DATASETS[dataset]
    lab_dir = paths.SHARED_ROOT / "labels" / dataset / section / LABELER
    release = cxr_root / RELEASES[dataset]
    inputs = {
        "labels": artifacts.artifact_record(lab_dir),
        "sources": [artifacts.file_record(cxr_root / f) for f in adapter.SOURCE_FILES],
        "radgraph": artifacts.file_record(release),
    }
    config = {
        "stage": "sides",
        "dataset": dataset,
        "section": section,
        "labeler": LABELER,
        "graphs": f"RadGraph-XL release section_{section}.json, matched by text "
        "without case, whitespace or punctuation",
        "observations": list(radgraph_sides.PRESENT),
        "labels": radgraph_sides.LATERAL_LABELS,
        "side": "located_at anatomy, its modifiers, and the anatomy of observations "
        "suggestive_of it; left and right = bilateral",
    }

    def build(out: Path) -> dict:
        with zipfile.ZipFile(release) as z:
            graphs = json.load(z.open(f"section_{section}.json"))
        by_text = {}
        for g in graphs:
            (entry,) = g.values()
            by_text.setdefault(radgraph_sides.normalize(entry["text"]), entry)
        studies = adapter.load_studies(cxr_root).set_index("study_id")
        labels = artifacts.read_table(lab_dir / "labels.csv").set_index("study_id")
        rows, unmatched = [], 0
        for study_id, y in labels.iterrows():
            graph = by_text.get(radgraph_sides.normalize(studies.at[study_id, section]))
            if graph is None:
                unmatched += 1
                continue
            sides = radgraph_sides.label_sides(graph["entities"])
            for name, side in sides.items():
                if y[name] in (1, -1):
                    rows.append((study_id, name, side))
        table = pd.DataFrame(rows, columns=["study_id", "label", "side"])
        table.to_csv(out / "sides.csv", index=False)
        mentioned = labels[list(radgraph_sides.LATERAL_LABELS)].isin([1, -1])
        return {
            "n_studies": len(labels),
            "n_unmatched": unmatched,
            "n_sided": len(table),
            "sided_share_of_mentions": round(
                len(table) / int(mentioned.values.sum()), 4
            ),
        }

    return artifacts.publish(
        f"sides/{dataset}/{section}/{LABELER}", config, inputs, build, overwrite
    )
