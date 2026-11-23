"""Stage ``severity``: the size or severity of each study's mentioned labels, by rule.

Writes ``severity/<dataset>/<section>/<labeler>/severity.csv``: study_id, label,
severity, one row per label the labeler mentions (positive or uncertain) whose size or
severity the section's text states (labelers/severity.py: "small", "mild to moderate").
Section ``findings+impression``: the joined text. No GPU.
"""

from pathlib import Path

import pandas as pd

from labrag import artifacts, paths
from labrag.data import DATASETS, JOINT, add_joint, check_section
from labrag.labelers import severity

LABELER = "chexbert14"


def relpath(dataset: str, section: str) -> str:
    """The severity artifact of a dataset's section."""
    return f"severity/{dataset}/{section}/{LABELER}"


def run(
    dataset: str,
    section: str,
    overwrite: bool = False,
    cxr_root: Path | None = None,
) -> Path:
    check_section("severity", dataset, section, joint=True)
    cxr_root = cxr_root or paths.CXR_ROOT
    adapter = DATASETS[dataset]
    lab_dir = paths.SHARED_ROOT / "labels" / dataset / section / LABELER
    inputs = {
        "labels": artifacts.artifact_record(lab_dir),
        "sources": [artifacts.file_record(cxr_root / f) for f in adapter.SOURCE_FILES],
    }
    config = {
        "stage": "severity",
        "dataset": dataset,
        "section": section,
        "labeler": LABELER,
        "labels": list(severity.KEYWORDS),
        "severity": "nearest size or severity word to the label's keyword in a clause "
        "that doesn't negate it; distinct phrases in report order, at most "
        f"{severity.MAX_PHRASES}",
    }

    def build(out: Path) -> dict:
        studies = adapter.load_studies(cxr_root).set_index("study_id")
        if section == JOINT:
            studies = add_joint(studies)
        labels = artifacts.read_table(lab_dir / "labels.csv").set_index("study_id")
        names = [n for n in labels.columns if n in severity.KEYWORDS]
        mentioned = labels[names].isin([1, -1])
        rows = []
        for study_id, row in mentioned.iterrows():
            shown = [n for n in names if row[n]]
            if shown:
                found = severity.severities(studies.at[study_id, section], shown)
                rows += [(study_id, n, v) for n, v in found.items()]
        table = pd.DataFrame(rows, columns=["study_id", "label", "severity"])
        table.to_csv(out / "severity.csv", index=False)
        counts = table.label.value_counts()
        return {
            "n_studies": len(labels),
            "n_rows": len(table),
            "share_of_mentions": {
                n: round(int(counts.get(n, 0)) / max(1, int(mentioned[n].sum())), 4)
                for n in names
            },
        }

    return artifacts.publish(
        relpath(dataset, section), config, inputs, build, overwrite
    )
