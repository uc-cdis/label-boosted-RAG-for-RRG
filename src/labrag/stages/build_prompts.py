"""Stage ``build-prompts``: one chat prompt per target study, for one condition.

Writes ``prompts/<experiment>/<dataset>/<section>/<condition>/<prompt>/``
``prompts.jsonl.gz``: study_id, set, messages (one user message), retrieved (corpus ids,
best first, with image similarity and, for label ranking, Jaccard), labels (null
without), reference (trimmed only).

Conditions: ``retrieval-only`` (top K by image similarity) and ``labrag.<labeler>``
(Jaccard on positive labels, ties by image similarity; the target's predicted labels in
the prompt; corpus studies carry reference labels, binarized like the classifier). Its
classifier is ``classifiers/<dataset>/<encoder>.<classifier>``: ``lr`` by default, any
other written to ``labrag.<classifier>.<labeler>``.
``labrag-oracle.<labeler>`` is labrag with the target's reference labels instead of
predicted ones: the ceiling for any classifier. Two diagnostics between the two (they
read the reference, so they are never a method): ``labrag-drop-fp.<labeler>``, the
predicted labels without their false positives, and ``labrag-add-fn.<labeler>``, with
their misses added. ``no-context``: no retrieval and no labels, so its prompt must show
the image. Retrieval never returns a study of the target's patient.

Predicted corpus labels, for labrag only: ``--corpus-labels predicted`` takes them from
the classifier's out-of-fold predictions (train role; a ``-oofK`` classifier) and its
plain predictions (val role), directory suffix ``.corpus-predicted``.
``--predictions <source>/<name>/<labeler>`` takes target and corpus labels from a
predict artifact (a classifier from another dataset, e.g. PadChest-60), written to
``labrag.<source>.<classifier>.<labeler>``.

A prompt version with {image} shows the study's preferred image (the one embedded; its
path relative to the CXR root in the record's ``image``).
"""

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from labrag import artifacts, paths, prompts, retrieval
from labrag.data import DATASETS
from labrag.stages.classify import BINARIZE, LABELER_SUFFIX, POSITIVE

# experiment -> dataset, target roles, corpus roles
EXPERIMENTS = {
    "pilot": {
        "dataset": "chexpertplus",
        "targets": ("pilot", "prompt_tuning"),
        "corpus": ("train", "val"),
    },
}
# label diagnostics -> how the target's labels mix predicted and reference
DIAGNOSTICS = {
    "labrag-drop-fp": "predicted and reference",
    "labrag-add-fn": "predicted or reference",
}
CONDITIONS = ("no-context", "retrieval-only", "labrag", "labrag-oracle", *DIAGNOSTICS)
DEFAULT_PROMPT = {
    "no-context": "section-image",
    "retrieval-only": "preprint-naive",
    "labrag": "preprint-simple",
    "labrag-oracle": "preprint-simple",
} | {c: "preprint-simple" for c in DIAGNOSTICS}
K = 5
ENCODER = "raddino"
CLASSIFIER = "lr"  # default; the others name their condition directory
LABELER = "chexbert14"


def run(
    experiment: str,
    section: str,
    condition: str,
    prompt: str | None = None,
    classifier: str = CLASSIFIER,
    uncertain: str = "zeros",
    corpus_labels: str = "reference",
    predictions: str | None = None,
    overwrite: bool = False,
    cxr_root: Path | None = None,
) -> Path:
    cxr_root = cxr_root or paths.CXR_ROOT
    exp = EXPERIMENTS[experiment]
    dataset = exp["dataset"]
    adapter = DATASETS[dataset]
    prompt = prompt or DEFAULT_PROMPT[condition]
    version = prompts.load(prompt)
    with_labels = condition.startswith("labrag")
    with_examples = condition != "no-context"
    with_image = prompts.uses_image(version)
    oracle = condition == "labrag-oracle"
    predicted = condition == "labrag" or condition in DIAGNOSTICS
    if classifier != CLASSIFIER and not predicted:
        raise SystemExit("build-prompts: --classifier is for predicted labels")
    if (corpus_labels != "reference" or predictions) and condition != "labrag":
        raise SystemExit("build-prompts: predicted corpus labels are for labrag only")
    if predictions and (classifier != CLASSIFIER or corpus_labels != "reference"):
        raise SystemExit("build-prompts: --predictions brings its own labels")
    if corpus_labels == "predicted" and "-oof" not in classifier:
        raise SystemExit(
            "build-prompts: --corpus-labels predicted needs a -oofK classifier"
        )
    transfer = predictions is not None
    if (prompts.uses_labels(version), prompts.uses_examples(version)) != (
        with_labels,
        with_examples,
    ):
        raise SystemExit(f"prompt {prompt!r} does not fit condition {condition!r}")

    labeler = LABELER + LABELER_SUFFIX[uncertain]
    emb_dir = paths.SHARED_ROOT / "embeddings" / dataset / ENCODER
    lab_dir = paths.SHARED_ROOT / "labels" / dataset / section / LABELER
    clf_dir = (
        paths.SHARED_ROOT
        / "classifiers"
        / dataset
        / f"{ENCODER}.{classifier}"
        / section
        / labeler
    )
    inputs = {
        "labels": artifacts.artifact_record(lab_dir),
        "sources": [artifacts.file_record(cxr_root / f) for f in adapter.SOURCE_FILES],
        "splits": [
            artifacts.file_record(paths.SPLITS_DIR / f) for f in adapter.SPLIT_FILES
        ],
    }
    config = {
        "stage": "build-prompts",
        "experiment": experiment,
        "dataset": dataset,
        "section": section,
        "targets": list(exp["targets"]),
        "corpus": list(exp["corpus"]),
        "condition": condition,
        "prompt": {"name": prompt, **version},
    }
    if with_examples:
        inputs["embeddings"] = artifacts.artifact_record(emb_dir)
        config |= {
            "k": K,
            "encoder": ENCODER,
            "similarity": "cosine",
            "report_text": "whitespace runs collapsed to one space",
        }
    if with_image:
        image_root = cxr_root / adapter.IMAGE_ROOT
        inputs["images"] = {"root": str(image_root)}  # too many to hash
        config["image"] = "preferred per study (the embedded one)"
    if transfer:
        pred_dir = paths.SHARED_ROOT / "predictions" / dataset / section
        source, clf_name, labeler = predictions.split("/")
        pred_dir = pred_dir / f"{source}.{clf_name}.{labeler}"
        inputs["predictions"] = artifacts.artifact_record(pred_dir)
        config["predictions"] = predictions
    elif predicted:
        inputs["classifier"] = artifacts.artifact_record(clf_dir)
        clf_inputs = artifacts.read_manifest(clf_dir)["inputs"]
        for name in ("embeddings", "labels"):  # the classifier saw these same inputs
            if name == "embeddings" and clf_inputs[name]["path"] != str(emb_dir):
                continue  # trained on tokens: its own image check is in classify
            assert (
                clf_inputs[name]["outputs"] == inputs[name]["outputs"]
            ), f"classifier {name} differ"
        config["classifier"] = classifier
    if with_labels:
        config |= {
            "labeler": labeler,
            "target_labels": (
                "reference" if oracle else DIAGNOSTICS.get(condition, "predicted")
            ),
            "corpus_labels": (
                f"predicted by {predictions}"
                if transfer
                else (
                    "predicted: out of fold (train), plain (val)"
                    if corpus_labels == "predicted"
                    else "reference"
                )
            ),
            "binarize": "predicted 0/1" if transfer else BINARIZE[uncertain],
            "ranking": "jaccard of positive labels (empty vs empty = 1), then cosine",
        }
        condition_dir = f"{condition}.{labeler}"
        if classifier != CLASSIFIER:
            condition_dir = f"{condition}.{classifier}.{labeler}"
        if corpus_labels == "predicted":
            condition_dir += ".corpus-predicted"
        if transfer:
            short = clf_name.removeprefix(f"{ENCODER}.")
            condition_dir = f"{condition}.{source}.{short}.{labeler}"
    else:
        if with_examples:
            config["ranking"] = "cosine"
        condition_dir = condition

    def build(out: Path) -> dict:
        labels = artifacts.read_table(lab_dir / "labels.csv").set_index("study_id")
        names = list(labels.columns)
        if transfer:
            transferred = artifacts.read_table(pred_dir / "predictions.csv")
            transferred = transferred.set_index("study_id")
            names = list(transferred.columns)
        studies = adapter.load_studies(cxr_root).set_index("study_id")
        studies["role"] = adapter.classifier_roles(studies, paths.SPLITS_DIR)
        studies = studies.loc[labels.index]  # the studies with this section
        target = studies[studies.role.isin(exp["targets"])]
        corpus = studies[studies.role.isin(exp["corpus"])]
        if not with_examples:
            summary = write(out, [record(i, row) for i, row in target.iterrows()])
            print(json.dumps(summary, indent=2))
            return summary

        embedded = artifacts.read_table(emb_dir / "studies.csv").set_index("study_id")
        embedded["row"] = np.arange(len(embedded))
        emb = np.load(emb_dir / "embeddings.npy")

        def embeddings(rows: pd.DataFrame) -> np.ndarray:
            # The embedded image must be the one the adapter picks for the study today.
            e = embedded.loc[rows.index]
            assert (e.image_path == rows.image_path).all(), "image mismatch"
            return emb[e.row.to_numpy()]

        sim = retrieval.cosine(embeddings(target), embeddings(corpus))
        exclude = target.subject_id.to_numpy()[:, None] == corpus.subject_id.to_numpy()
        if with_labels:

            def reference(rows: pd.DataFrame) -> np.ndarray:
                y = labels.loc[rows.index, names].isin(POSITIVE[uncertain])
                return y.to_numpy(dtype=int)

            if transfer:
                corpus_y = transferred.loc[corpus.index].to_numpy(dtype=int)
            elif corpus_labels == "predicted":
                corpus_y = predicted_corpus(clf_dir, corpus, names)
            else:
                corpus_y = reference(corpus)
            if transfer:
                target_y = transferred.loc[target.index].to_numpy(dtype=int)
            elif oracle:
                target_y = reference(target)
            else:
                pred = artifacts.read_table(clf_dir / "predictions.csv").set_index(
                    "study_id"
                )
                assert list(pred.columns) == names, "classifier labels differ"
                target_y = pred.loc[target.index, names].to_numpy(dtype=int)
                if condition in DIAGNOSTICS:
                    target_y = mix_labels(condition, target_y, reference(target))
            overlap = retrieval.jaccard(target_y, corpus_y)
        else:
            overlap = None
        top = retrieval.top_k(K, sim, exclude, overlap)

        def positives(row: np.ndarray) -> list[str]:
            return [n for n, v in zip(names, row) if v]

        reports = corpus[section].to_numpy()
        corpus_ids = corpus.index.to_numpy()
        records = []
        for i, (study_id, row) in enumerate(target.iterrows()):
            idx = top[i]
            examples = [
                (reports[j], positives(corpus_y[j]) if with_labels else []) for j in idx
            ]
            retrieved = {
                "retrieved": corpus_ids[idx].tolist(),
                "retrieved_similarity": sim[i, idx].tolist(),
            }
            if with_labels:
                retrieved["retrieved_jaccard"] = overlap[i, idx].tolist()
            labels_i = positives(target_y[i]) if with_labels else None
            records.append(record(study_id, row, examples, labels_i, retrieved))
        summary = write(out, records)
        summary["n_corpus"] = len(corpus)
        if with_labels:
            best = overlap[np.arange(len(top)), top[:, 0]]
            summary["top1_jaccard_1"] = float((best == 1).mean())
            summary["mean_retrieved_jaccard"] = float(
                np.take_along_axis(overlap, top, 1).mean()
            )
        print(json.dumps(summary, indent=2))
        return summary

    def record(study_id, row, examples=(), labels_i=None, retrieved=None) -> dict:
        r = {
            "study_id": study_id,
            "set": row.role,
            "messages": prompts.render(version, list(examples), labels_i, section),
        }
        if with_image:
            r["image"] = f"{adapter.IMAGE_ROOT}/{row.image_path}"
        return r | (retrieved or {}) | {"labels": labels_i, "reference": row[section]}

    def write(out: Path, records: list[dict]) -> dict:
        # mtime=0 and no file name in the gzip header: reruns are byte-identical.
        with open(out / "prompts.jsonl.gz", "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
                for r in records:
                    f.write((json.dumps(r, ensure_ascii=False) + "\n").encode())

        chars = [prompts.text_chars(r["messages"]) for r in records]
        sets = pd.Series([r["set"] for r in records]).value_counts().sort_index()
        return {
            "n_targets": len(records),
            "n_by_set": sets.to_dict(),
            "prompt_chars": {"mean": float(np.mean(chars)), "max": int(max(chars))},
        }

    return artifacts.publish(
        f"prompts/{experiment}/{dataset}/{section}/{condition_dir}/{prompt}",
        config,
        inputs,
        build,
        overwrite,
    )


def predicted_corpus(clf_dir: Path, corpus: pd.DataFrame, names: list[str]):
    """Corpus labels from a -oofK classifier: out of fold for its train role, plain
    predictions for the rest (val: never trained on)."""
    oof = artifacts.read_table(clf_dir / "oof_predictions.csv").set_index("study_id")
    plain = artifacts.read_table(clf_dir / "predictions.csv").set_index("study_id")
    assert list(oof.columns) == list(plain.columns) == names, "classifier labels differ"
    train = corpus.role.eq("train")
    y = plain.loc[corpus.index, names].copy()
    y.loc[train] = oof.loc[corpus.index[train], names]
    return y.to_numpy(dtype=int)


def mix_labels(condition: str, predicted: np.ndarray, reference: np.ndarray):
    """A diagnostic's target labels (0/1 per study and label)."""
    if condition == "labrag-drop-fp":
        return predicted & reference
    if condition == "labrag-add-fn":
        return predicted | reference
    raise ValueError(condition)
