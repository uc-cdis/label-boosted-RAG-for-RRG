"""Stage ``build-prompts``: one chat prompt per target study, for one condition.

Writes ``prompts/<experiment>/<dataset>/<section>/<condition>/<prompt>/prompts.jsonl.gz``,
one JSON record per target study:
- ``study_id``, ``set`` (the target's role, e.g. pilot or prompt_tuning);
- ``messages``: the chat messages (a single user message; generate adds any
  model-specific system prompt);
- ``retrieved``: the corpus study_ids in the prompt, best first, with their
  ``retrieved_similarity`` (image cosine) and, for label ranking, ``retrieved_jaccard``;
- ``labels``: the target's positive labels in the prompt (null without labels);
- ``reference``: the target's reference section text (whitespace-trimmed only).

Conditions (docs/revision-plan.md §3), so far:
- ``retrieval-only``: the K corpus studies most similar by image; no labels.
- ``labrag.<labeler>``: corpus ranked by Jaccard overlap of positive labels, then by
  image similarity; the labels in the prompt. The target's labels are the classifier's
  predictions (``classifiers/<dataset>/<encoder>.lr/<section>/<labeler>/``); corpus
  studies carry their reference labels, binarized with the same uncertain policy
  (``<labeler>`` ends in ``.u0`` for U-Zeros).

Corpus and targets come from the experiment's roles (the adapter's
``classifier_roles``). Retrieval never returns a study of the target's patient.
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
CONDITIONS = ("retrieval-only", "labrag")
DEFAULT_PROMPT = {"retrieval-only": "preprint-naive", "labrag": "preprint-simple"}
K = 5
ENCODER = "raddino"
CLASSIFIER = "lr"
LABELER = "chexbert14"


def run(
    experiment: str,
    section: str,
    condition: str,
    prompt: str | None = None,
    uncertain: str = "ones",
    overwrite: bool = False,
    cxr_root: Path | None = None,
) -> Path:
    cxr_root = cxr_root or paths.CXR_ROOT
    exp = EXPERIMENTS[experiment]
    dataset = exp["dataset"]
    adapter = DATASETS[dataset]
    prompt = prompt or DEFAULT_PROMPT[condition]
    version = prompts.load(prompt)
    with_labels = condition == "labrag"
    if prompts.uses_labels(version) != with_labels:
        raise SystemExit(f"prompt {prompt!r} does not fit condition {condition!r}")

    labeler = LABELER + LABELER_SUFFIX[uncertain]
    emb_dir = paths.SHARED_ROOT / "embeddings" / dataset / ENCODER
    lab_dir = paths.SHARED_ROOT / "labels" / dataset / section / LABELER
    clf_dir = (
        paths.SHARED_ROOT
        / "classifiers"
        / dataset
        / f"{ENCODER}.{CLASSIFIER}"
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
    if with_labels:
        inputs["classifier"] = artifacts.artifact_record(clf_dir)
        for name in ("embeddings", "labels"):  # the classifier saw these same inputs
            clf_input = artifacts.read_manifest(clf_dir)["inputs"][name]["outputs"]
            assert clf_input == inputs[name]["outputs"], f"classifier {name} differ"
        config |= {
            "labeler": labeler,
            "classifier": CLASSIFIER,
            "target_labels": "predicted",
            "corpus_labels": "reference",
            "binarize": BINARIZE[uncertain],
            "ranking": "jaccard of positive labels (empty vs empty = 1), then cosine",
        }
        condition_dir = f"{condition}.{labeler}"
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
            corpus_y = labels.loc[corpus.index, names].isin(POSITIVE[uncertain])
            corpus_y = corpus_y.to_numpy(dtype=int)
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
