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
predicted ones: the ceiling for any classifier. Retrieval never returns a study of the
target's patient.
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
CONDITIONS = ("retrieval-only", "labrag", "labrag-oracle")
DEFAULT_PROMPT = {
    "retrieval-only": "preprint-naive",
    "labrag": "preprint-simple",
    "labrag-oracle": "preprint-simple",
}
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
    overwrite: bool = False,
    cxr_root: Path | None = None,
) -> Path:
    cxr_root = cxr_root or paths.CXR_ROOT
    exp = EXPERIMENTS[experiment]
    dataset = exp["dataset"]
    adapter = DATASETS[dataset]
    prompt = prompt or DEFAULT_PROMPT[condition]
    version = prompts.load(prompt)
    with_labels = condition != "retrieval-only"
    oracle = condition == "labrag-oracle"
    if classifier != CLASSIFIER and condition != "labrag":
        raise SystemExit("build-prompts: --classifier is for --condition labrag")
    if prompts.uses_labels(version) != with_labels:
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
        "embeddings": artifacts.artifact_record(emb_dir),
        "labels": artifacts.artifact_record(lab_dir),
        "sources": [artifacts.file_record(cxr_root / adapter.SOURCE_CSV)],
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
        "k": K,
        "encoder": ENCODER,
        "similarity": "cosine",
        "prompt": {"name": prompt, **version},
        "report_text": "whitespace runs collapsed to one space",
    }
    if with_labels and not oracle:
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
            "target_labels": "reference" if oracle else "predicted",
            "corpus_labels": "reference",
            "binarize": BINARIZE[uncertain],
            "ranking": "jaccard of positive labels (empty vs empty = 1), then cosine",
        }
        condition_dir = f"{condition}.{labeler}"
        if classifier != CLASSIFIER:
            condition_dir = f"{condition}.{classifier}.{labeler}"
    else:
        config["ranking"] = "cosine"
        condition_dir = condition

    def build(out: Path) -> dict:
        labels = pd.read_csv(lab_dir / "labels.csv").set_index("study_id")
        names = list(labels.columns)
        studies = adapter.load_studies(cxr_root).set_index("study_id")
        studies["role"] = adapter.classifier_roles(studies, paths.SPLITS_DIR)
        studies = studies.loc[labels.index]  # the studies with this section
        target = studies[studies.role.isin(exp["targets"])]
        corpus = studies[studies.role.isin(exp["corpus"])]

        embedded = pd.read_csv(emb_dir / "studies.csv").set_index("study_id")
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

            corpus_y = reference(corpus)
            if oracle:
                target_y = reference(target)
            else:
                pred = pd.read_csv(clf_dir / "predictions.csv").set_index("study_id")
                assert list(pred.columns) == names, "classifier labels differ"
                target_y = pred.loc[target.index, names].to_numpy(dtype=int)
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
            labels_i = positives(target_y[i]) if with_labels else None
            record = {
                "study_id": study_id,
                "set": row.role,
                "messages": prompts.render(version, examples, labels_i),
                "retrieved": corpus_ids[idx].tolist(),
                "retrieved_similarity": sim[i, idx].tolist(),
            }
            if with_labels:
                record["retrieved_jaccard"] = overlap[i, idx].tolist()
            record |= {"labels": labels_i, "reference": row[section]}
            records.append(record)

        # mtime=0 and no file name in the gzip header: reruns are byte-identical.
        with open(out / "prompts.jsonl.gz", "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
                for record in records:
                    f.write((json.dumps(record, ensure_ascii=False) + "\n").encode())

        chars = [len(r["messages"][0]["content"]) for r in records]
        summary = {
            "n_targets": len(target),
            "n_by_set": target.role.value_counts().sort_index().to_dict(),
            "n_corpus": len(corpus),
            "prompt_chars": {"mean": float(np.mean(chars)), "max": int(max(chars))},
        }
        if with_labels:
            best = overlap[np.arange(len(top)), top[:, 0]]
            summary["top1_jaccard_1"] = float((best == 1).mean())
            summary["mean_retrieved_jaccard"] = float(
                np.take_along_axis(overlap, top, 1).mean()
            )
        print(json.dumps(summary, indent=2))
        return summary

    return artifacts.publish(
        f"prompts/{experiment}/{dataset}/{section}/{condition_dir}/{prompt}",
        config,
        inputs,
        build,
        overwrite,
    )
