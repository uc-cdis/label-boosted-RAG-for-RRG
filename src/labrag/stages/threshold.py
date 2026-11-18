"""Stage ``threshold``: a classifier's predictions under another threshold rule.

``--classifier`` is a classify artifact under ``classifiers/``. Reads its probabilities
(probs.<section>.csv) and the labels it trained on; writes the same path with the
rule's suffix on the classifier name: predictions.<section>.csv and metrics.csv,
thresholds per section and label on its val studies, the source's binarization and
consistency rule. No model of its own: build-prompts checks the source's inputs, predict refuses it.

Rules: ``0.5`` for every label; ``prevalence``: per label, as many val predictions as
val positives (more only on ties). Max F1 is classify's own.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from labrag import artifacts, paths
from labrag.classifiers import logistic
from labrag.stages import classify

RULES = {"0.5": "-t50", "prevalence": "-prev"}  # rule: classifier name suffix
DESCRIPTION = {
    "0.5": "0.5 for every label",
    "prevalence": "per label, the k-th highest val probability, k = val positives",
    "max-f1": "max F1",
}


def run(classifier: str, rule: str, overwrite: bool = False) -> Path:
    src_rel = f"classifiers/{classifier.strip('/').removeprefix('classifiers/')}"
    src = paths.SHARED_ROOT / src_rel
    manifest = artifacts.read_manifest(src)
    if manifest is None or manifest["config"]["stage"] != "classify":
        raise SystemExit(f"threshold: {src_rel} is not a classify artifact")
    src_config = manifest["config"]
    if "oof" in src_config:
        raise SystemExit(
            "threshold: -oofK classifiers (fold models' thresholds) unsupported"
        )
    parts = src_rel.split("/")  # classifiers, dataset, name, section, labeler
    relpath = "/".join([*parts[:2], parts[2] + RULES[rule], *parts[3:]])

    labels = {}
    for s, record in manifest["inputs"]["labels"].items():
        now = artifacts.artifact_record(Path(record["path"]))
        if now["outputs"] != record["outputs"]:
            raise SystemExit(f"threshold: {record['path']} changed since classify")
        labels[s] = now
    inputs = {"classifier": artifacts.artifact_record(src), "labels": labels}
    config = {
        "stage": "threshold",
        "classifier": src_rel,
        "rule": rule,
        "threshold": f"{DESCRIPTION[rule]} on val; predict prob >= threshold",
        "binarize": src_config["binarize"],
        "consistency": src_config["consistency"],
    }
    ones = src_config["binarize"] == classify.BINARIZE["ones"]
    positive = classify.POSITIVE["ones" if ones else "zeros"]
    release = src_config["binarize"] == classify.RELEASE_BINARIZE

    def build(out: Path) -> dict:
        metrics, summary = [], {}
        for s, record in labels.items():
            probs = artifacts.read_table(src / f"probs.{s}.csv").set_index("study_id")
            ref = artifacts.read_table(Path(record["path"]) / "labels.csv")
            ref = ref.set_index("study_id")
            names = list(ref.columns)
            assert list(probs.columns.drop("role")) == names, "labels differ"
            P, Y, role, has = matrices(probs, ref, names, positive)
            val, official = (role == "val") & has, (role == "official_valid") & has
            cutoff = cutoffs(rule, Y[val], P[val])
            pred = classify.predict_labels(P, cutoff, names, release)
            preds = pd.DataFrame(pred, columns=names)
            preds.insert(0, "study_id", probs.index)
            preds.to_csv(out / f"predictions.{s}.csv", index=False)
            table = pd.DataFrame(
                {
                    "section": s,
                    "label": names,
                    "train_pos": Y[(role == "train") & has].sum(0),
                    "threshold": cutoff,
                    **classify._scores("val", Y[val], P[val], pred[val]),
                    **(
                        classify._scores(
                            "official_valid", Y[official], P[official], pred[official]
                        )
                        if official.any()
                        else {}
                    ),
                }
            )
            metrics.append(table)
            summary[s] = {
                "val_studies": int(val.sum()),
                "val_predicted_per_study": float(pred[val].sum(1).mean()),
                "val_reference_per_study": float(Y[val].sum(1).mean()),
                "val_wrong_per_study": float((pred[val] > Y[val]).sum(1).mean()),
                "val_missed_per_study": float((pred[val] < Y[val]).sum(1).mean()),
            }
        metrics = pd.concat(metrics, ignore_index=True)
        metrics.to_csv(out / "metrics.csv", index=False)
        print(metrics.to_string(index=False))
        print(summary)
        return {"val": summary}

    return artifacts.publish(relpath, config, inputs, build, overwrite)


def matrices(probs: pd.DataFrame, ref: pd.DataFrame, names: list[str], positive):
    """Probabilities, 0/1 reference labels (0 where unlabelled), roles, and whether
    each row of ``probs`` has labels."""
    has = probs.index.isin(ref.index)
    Y = np.zeros((len(probs), len(names)), dtype=int)
    Y[has] = ref.loc[probs.index[has], names].isin(positive).to_numpy(dtype=int)
    return probs[names].to_numpy(), Y, probs.role.to_numpy(), has


def cutoffs(rule: str, Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    """One threshold per label column, from val labels ``Y`` and probabilities ``P``."""
    if rule == "0.5":
        return np.full(Y.shape[1], 0.5)
    pick = {
        "prevalence": logistic.prevalence_threshold,
        "max-f1": logistic.f1_threshold,
    }
    return np.array([pick[rule](y, p) for y, p in zip(Y.T, P.T)])


def trained_manifest(clf_dir: Path) -> dict:
    """The classify manifest behind ``clf_dir`` (itself, or a threshold's source)."""
    manifest = artifacts.read_manifest(clf_dir)
    if manifest["config"]["stage"] == "threshold":
        return artifacts.read_manifest(
            paths.SHARED_ROOT / manifest["config"]["classifier"]
        )
    return manifest
