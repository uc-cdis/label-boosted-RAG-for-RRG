"""Stage ``build-prompts``: one chat prompt per target study, for one condition.

Writes ``prompts/<experiment>/<dataset>/<section>/<condition>/<prompt>/``
``prompts.jsonl.gz``: study_id, set, messages (one user message), retrieved (corpus ids,
best first, with image similarity and, for label ranking, Jaccard), labels (null
without), reference (trimmed only).

An experiment fixes, per dataset, the target and corpus roles (EXPERIMENTS): ``pilot``
(CheXpert Plus pilot split), ``val`` (every val study of MIMIC-CXR and ReXGradient,
corpus train only; their thresholds were set on val, so max-F1 labels are in-sample),
``smoke`` (100 of those val patients, one study each).

Conditions: ``retrieval-only`` (top K by image similarity) and ``labrag.<labeler>``
(Jaccard on positive labels, ties by image similarity; the target's predicted labels in
the prompt; corpus studies carry reference labels, binarized like the classifier).
``--classifier <dataset>/<name>/<sections>/<labeler>`` (under ``classifiers/``; the
experiment's dataset, the --uncertain labeler) gives its predictions.<section>.csv,
written to ``labrag.<name>.<sections>.<labeler>`` (name without the encoder).
``labrag-oracle.<labeler>`` is labrag with the target's reference labels instead of
predicted ones: the ceiling for any classifier. Two diagnostics between the two (they
read the reference, so they are never a method): ``labrag-drop-fp.<labeler>``, the
predicted labels without their false positives, and ``labrag-add-fn.<labeler>``, with
their misses added. ``no-context``: no retrieval and no labels, so its prompt must show
the image. Retrieval never returns a study of the target's patient.

``--predictions <source>/<name>/<sections>/<labeler>`` takes the target's labels from a
predict artifact (a classifier from another dataset or fine-tuned on images: its
readout for the section, or its only one), written to
``labrag.<source>.<name>.<sections>.<labeler>``.
Corpus labels are the reference by default. ``--corpus-labels predicted`` (labrag only,
directory suffix ``.corpus-predicted``) takes them from the classifier: with
``--classifier``, its out-of-fold predictions (train role; a ``-oofK`` classifier) and
plain ones (val role); with ``--predictions``, the predict artifact (needed when the
labeler has no reference labels here, e.g. PadChest-60).

``--ranking image`` (label conditions): the same top K by image similarity as
retrieval-only, the labels shown but not used to rank; directory suffix
``.image-ranked``. It separates the label list in the prompt from label-based retrieval.
``--sides`` (labrag-oracle): each label shown with its side from the ``sides`` stage,
"Pleural Effusion (left)", for the target and the examples; ranking unchanged;
directory suffix ``.sides``.
``--confidence`` (labrag): each target label shown with the classifier's probability,
"Atelectasis (28%)" (its probs file); example labels, label set and ranking unchanged;
directory suffix ``.confidence``.
Multiclass predictions (``--predictions .../chexbert14.mc``, finetune --multiclass)
give CheXbert's four values per target label; their positives rank as usual. A prompt
version with ``label_categories`` (e.g. [Positive, Negative, Uncertain, Unmentioned])
shows the labels under those categories, for the target (predicted) and the examples
(their reference values); any other version shows the positives only. A version with
{absent} also lists the target's labels predicted negative (multiclass only; examples
show their positives as usual).

A prompt version with {image} shows the study's preferred image (the one embedded; its
path relative to the CXR root in the record's ``image``). A version with {indication}
shows the target's indication (the adapter's ``indication``; also in the record), any
condition; the examples show none.
"""

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from labrag import artifacts, paths, prompts, retrieval
from labrag.data import DATASETS, JOIN
from labrag.labelers import radgraph_sides
from labrag.stages import threshold
from labrag.stages.classify import BINARIZE, LABELER_SUFFIX, MULTICLASS_SUFFIX, POSITIVE

# experiment -> dataset -> target roles, corpus roles and, optionally, a sample of the
# targets: that many patients drawn with the seed, one study each
VAL = {
    "targets": ("val",),  # test stays unseen until the generation config is frozen
    "corpus": ("train",),
}
SMOKE = VAL | {"sample": {"patients": 100, "seed": 20261119}}
EXPERIMENTS = {
    "pilot": {
        "chexpertplus": {
            "targets": ("pilot", "prompt_tuning"),
            "corpus": ("train", "val"),
        },
    },
    "smoke": {"mimic": SMOKE, "rexgradient": SMOKE},
    "val": {"mimic": VAL, "rexgradient": VAL},
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
LABELER = "chexbert14"


def run(
    experiment: str,
    dataset: str,
    section: str,
    condition: str,
    prompt: str | None = None,
    classifier: str | None = None,
    uncertain: str = "zeros",
    corpus_labels: str = "reference",
    predictions: str | None = None,
    ranking: str = "labels",
    sides: bool = False,
    confidence: bool = False,
    overwrite: bool = False,
    cxr_root: Path | None = None,
) -> Path:
    cxr_root = cxr_root or paths.CXR_ROOT
    if dataset not in EXPERIMENTS[experiment]:
        raise SystemExit(
            f"build-prompts: {experiment} runs on {sorted(EXPERIMENTS[experiment])}"
        )
    exp = EXPERIMENTS[experiment][dataset]
    adapter = DATASETS[dataset]
    prompt = prompt or DEFAULT_PROMPT[condition]
    version = prompts.load(prompt)
    with_labels = condition.startswith("labrag")
    with_examples = condition != "no-context"
    with_image = prompts.uses_image(version)
    oracle = condition == "labrag-oracle"
    predicted = condition == "labrag" or condition in DIAGNOSTICS
    if (classifier is not None or predictions) and not predicted:
        raise SystemExit(
            "build-prompts: --classifier/--predictions are for predicted labels"
        )
    if corpus_labels != "reference" and condition != "labrag":
        raise SystemExit("build-prompts: predicted corpus labels are for labrag only")
    if predictions and classifier is not None:
        raise SystemExit("build-prompts: --predictions brings its own labels")
    if predicted and not predictions and classifier is None:
        raise SystemExit(
            "build-prompts: predicted labels need --classifier "
            "<dataset>/<name>/<sections>/<labeler> or --predictions"
        )
    if corpus_labels == "predicted" and classifier and "-oof" not in classifier:
        raise SystemExit(
            "build-prompts: --corpus-labels predicted needs a -oofK classifier"
        )
    if ranking != "labels" and not with_labels:
        raise SystemExit("build-prompts: --ranking is for conditions with labels")
    if sides and condition != "labrag-oracle":
        raise SystemExit("build-prompts: --sides is for labrag-oracle")
    if confidence and condition != "labrag":
        raise SystemExit("build-prompts: --confidence is for labrag")
    transfer = predictions is not None
    categories = prompts.uses_categories(version)
    absent = prompts.uses_absent(version)
    indication = prompts.uses_indication(version)
    if (prompts.uses_labels(version), prompts.uses_examples(version)) != (
        with_labels,
        with_examples,
    ):
        raise SystemExit(f"prompt {prompt!r} does not fit condition {condition!r}")

    labeler = LABELER + LABELER_SUFFIX[uncertain]
    multiclass = False
    if transfer:
        source, clf_name, clf_sections, labeler = parse_classifier(predictions)
        readout = section if section in clf_sections.split(JOIN) else clf_sections
        if JOIN in readout:
            raise SystemExit(f"build-prompts: {predictions} has no {section} readout")
        multiclass = labeler.endswith(MULTICLASS_SUFFIX)
        if multiclass and (
            condition != "labrag" or confidence or corpus_labels != "reference"
        ):
            raise SystemExit(
                "build-prompts: multiclass predictions are for labrag with reference "
                "corpus labels"
            )
        if (
            corpus_labels == "reference"
            and labeler.removesuffix(MULTICLASS_SUFFIX)
            != LABELER + LABELER_SUFFIX[uncertain]
        ):
            raise SystemExit(
                f"build-prompts: no {labeler} reference labels for the corpus; "
                "use --corpus-labels predicted"
            )
    elif predicted:
        source, clf_name, clf_sections, clf_labeler = parse_classifier(classifier)
        if source != dataset:
            raise SystemExit(
                f"build-prompts: a {source} classifier on {dataset} goes through "
                "predict and --predictions"
            )
        if clf_labeler != labeler:
            raise SystemExit(
                f"build-prompts: --uncertain gives {labeler}, not {clf_labeler}"
            )
        if section not in clf_sections.split(JOIN):
            raise SystemExit(f"build-prompts: {classifier} has no {section} readout")
    if categories and not multiclass:
        raise SystemExit(
            f"build-prompts: prompt {prompt!r} shows label categories: it needs "
            "multiclass --predictions (chexbert14.mc)"
        )
    if absent and not multiclass:
        raise SystemExit(
            f"build-prompts: prompt {prompt!r} lists absent labels: it needs "
            "multiclass --predictions (chexbert14.mc)"
        )
    emb_dir = paths.SHARED_ROOT / "embeddings" / dataset / ENCODER
    lab_dir = paths.SHARED_ROOT / "labels" / dataset / section / LABELER
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
        **({"sample": exp["sample"]} if "sample" in exp else {}),
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
    if indication:
        config["indication"] = (
            f"target only: the adapter's indication, {prompts.NO_INDICATION!r} if none"
        )
    if with_image:
        image_root = cxr_root / adapter.IMAGE_ROOT
        inputs["images"] = {"root": str(image_root)}  # too many to hash
        config["image"] = "preferred per study (the embedded one)"
    if sides:
        sides_dir = paths.SHARED_ROOT / "sides" / dataset / section / LABELER
        inputs["sides"] = artifacts.artifact_record(sides_dir)
        if labeler != LABELER:
            raise SystemExit(f"build-prompts: sides are for {LABELER} labels")
    if transfer:
        pred_dir = paths.SHARED_ROOT / "predictions" / dataset / predictions.strip("/")
        inputs["predictions"] = artifacts.artifact_record(pred_dir)
        config["predictions"] = predictions
    elif predicted:
        clf_dir = paths.SHARED_ROOT / "classifiers" / classifier.strip("/")
        inputs["classifier"] = artifacts.artifact_record(clf_dir)
        clf_inputs = threshold.trained_manifest(clf_dir)["inputs"]
        for name in ("embeddings", "labels"):  # the classifier saw these same inputs
            record = clf_inputs[name]
            if name == "labels":
                record = record[section]  # one labels input per section
            if name == "embeddings" and record["path"] != str(emb_dir):
                continue  # trained on tokens: its own image check is in classify
            assert (
                record["outputs"] == inputs[name]["outputs"]
            ), f"classifier {name} differ"
        config["classifier"] = classifier
    if with_labels:
        config |= {
            "labeler": labeler,
            "target_labels": (
                "reference" if oracle else DIAGNOSTICS.get(condition, "predicted")
            ),
            "corpus_labels": (
                "reference"
                if corpus_labels == "reference"
                else (
                    f"predicted by {predictions}"
                    if transfer
                    else "predicted: out of fold (train), plain (val)"
                )
            ),
            "binarize": (
                "predicted 0/1"
                if transfer and corpus_labels == "predicted"
                else BINARIZE[uncertain]
            ),
            "ranking": (
                "jaccard of positive labels (empty vs empty = 1), then cosine"
                if ranking == "labels"
                else "cosine (labels shown, not used to rank)"
            ),
            **(
                {
                    "label_categories": version["label_categories"],
                    "label_values": "target predicted, examples reference",
                }
                if categories
                else {}
            ),
            **(
                {"absent": "target: labels predicted negative; examples: none"}
                if absent
                else {}
            ),
        }
        condition_dir = f"{condition}.{labeler}"
        if predicted:
            short = clf_name.removeprefix(f"{ENCODER}.")
            short = f"{source}.{short}" if transfer else short
            condition_dir = f"{condition}.{short}.{clf_sections}.{labeler}"
        if corpus_labels == "predicted":
            condition_dir += ".corpus-predicted"
        if ranking == "image":
            condition_dir += ".image-ranked"
        if sides:
            config["sides"] = "reference, target and examples: label (side)"
            condition_dir += ".sides"
        if confidence:
            config["confidence"] = "target labels: label (probability, % rounded, 1-99)"
            condition_dir += ".confidence"
    else:
        if with_examples:
            config["ranking"] = "cosine"
        condition_dir = condition

    def build(out: Path) -> dict:
        labels = artifacts.read_table(lab_dir / "labels.csv").set_index("study_id")
        names = list(labels.columns)
        if transfer:
            transferred = artifacts.read_table(pred_dir / f"predictions.{readout}.csv")
            transferred = transferred.set_index("study_id")
            names = list(transferred.columns)
        studies = adapter.load_studies(cxr_root).set_index("study_id")
        if indication and "indication" not in studies:
            raise SystemExit(f"build-prompts: no indication in {dataset}'s adapter")
        studies["role"] = adapter.classifier_roles(studies, paths.SPLITS_DIR)
        studies = studies.loc[labels.index]  # the studies with this section
        target = studies[studies.role.isin(exp["targets"])]
        if "sample" in exp:
            target = sample_targets(target, **exp["sample"])
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

            if corpus_labels == "reference":
                assert names == list(labels.columns), "classifier labels differ"
            if transfer and corpus_labels == "predicted":
                corpus_y = transferred.loc[predicted_rows(transferred, corpus)]
                corpus_y = corpus_y.to_numpy(dtype=int)
            elif corpus_labels == "predicted":
                corpus_y = predicted_corpus(clf_dir, section, corpus, names)
            else:
                corpus_y = reference(corpus)
            target_codes = None
            if transfer and multiclass:
                target_codes = transferred.loc[target.index, names].to_numpy()
                target_y = (target_codes == 1).astype(int)
            elif transfer:
                target_y = transferred.loc[target.index].to_numpy(dtype=int)
            elif oracle:
                target_y = reference(target)
            else:
                pred = artifacts.read_table(clf_dir / f"predictions.{section}.csv")
                pred = pred.set_index("study_id")
                assert list(pred.columns) == names, "classifier labels differ"
                target_y = pred.loc[target.index, names].to_numpy(dtype=int)
            if condition in DIAGNOSTICS:
                target_y = mix_labels(condition, target_y, reference(target))
            overlap = retrieval.jaccard(target_y, corpus_y)
        else:
            overlap = None
        top = retrieval.top_k(K, sim, exclude, overlap if ranking == "labels" else None)

        side = {}
        if sides:
            table = artifacts.read_table(sides_dir / "sides.csv")
            side = dict(zip(zip(table.study_id, table.label), table.side))

        prob = None
        if confidence:
            probs_file = (
                pred_dir / f"probs.{readout}.csv"
                if transfer
                else clf_dir / f"probs.{section}.csv"
            )
            prob = artifacts.read_table(probs_file).set_index("study_id")
            prob = prob.loc[target.index, names]

        if categories:
            corpus_codes = labels.loc[corpus.index, names].to_numpy()
            shown = version["label_categories"]

        def positives(row: np.ndarray, study_id: str, target=False) -> list[str]:
            shown = [n for n, v in zip(names, row) if v]
            if target and prob is not None:
                return [
                    prompts.with_probability(n, prob.at[study_id, n]) for n in shown
                ]
            return [radgraph_sides.sided(n, side.get((study_id, n))) for n in shown]

        reports = corpus[section].to_numpy()
        corpus_ids = corpus.index.to_numpy()
        records = []
        for i, (study_id, row) in enumerate(target.iterrows()):
            idx = top[i]
            examples = [
                (
                    reports[j],
                    (
                        prompts.categories(corpus_codes[j], names, shown)
                        if categories
                        else (
                            positives(corpus_y[j], corpus_ids[j]) if with_labels else []
                        )
                    ),
                )
                for j in idx
            ]
            retrieved = {
                "retrieved": corpus_ids[idx].tolist(),
                "retrieved_similarity": sim[i, idx].tolist(),
            }
            if with_labels:
                retrieved["retrieved_jaccard"] = overlap[i, idx].tolist()
            if categories:
                labels_i = prompts.categories(target_codes[i], names, shown)
            else:
                labels_i = (
                    positives(target_y[i], study_id, True) if with_labels else None
                )
            absent_i = (
                [n for n, c in zip(names, target_codes[i]) if c == 0]
                if absent
                else None
            )
            records.append(
                record(study_id, row, examples, labels_i, retrieved, absent_i)
            )
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

    def record(
        study_id, row, examples=(), labels_i=None, retrieved=None, absent_i=None
    ) -> dict:
        indication_i = None
        if indication:
            indication_i = "" if pd.isna(row.indication) else row.indication
        r = {
            "study_id": study_id,
            "set": row.role,
            "messages": prompts.render(
                version, list(examples), labels_i, section, absent_i, indication_i
            ),
        }
        if with_image:
            r["image"] = f"{adapter.IMAGE_ROOT}/{row.image_path}"
        shown = {"labels": labels_i} | ({"absent": absent_i} if absent else {})
        if indication:
            shown["indication"] = indication_i or None
        return r | (retrieved or {}) | shown | {"reference": row[section]}

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


def sample_targets(target: pd.DataFrame, patients: int, seed: int) -> pd.DataFrame:
    """``patients`` patients drawn uniformly with ``seed``, one random study each, in
    study order."""
    rng = np.random.default_rng(seed)
    picked = rng.choice(np.sort(target.subject_id.unique()), patients, replace=False)
    by_patient = target.groupby("subject_id").groups
    chosen = [rng.choice(np.sort(by_patient[p].to_numpy())) for p in picked]
    return target.loc[sorted(chosen)]


def parse_classifier(path: str) -> list[str]:
    """dataset, name, sections, labeler of a classifier path under classifiers/."""
    parts = path.strip("/").split("/")
    if len(parts) != 4:
        raise SystemExit(
            f"build-prompts: {path!r} is not <dataset>/<name>/<sections>/<labeler>"
        )
    return parts


def predicted_corpus(
    clf_dir: Path, section: str, corpus: pd.DataFrame, names: list[str]
):
    """Corpus labels from a -oofK classifier: out of fold for its train role, plain
    predictions for the rest (val: never trained on)."""
    oof = artifacts.read_table(clf_dir / f"oof_predictions.{section}.csv")
    oof = oof.set_index("study_id")
    plain = artifacts.read_table(clf_dir / f"predictions.{section}.csv")
    plain = plain.set_index("study_id")
    assert list(oof.columns) == list(plain.columns) == names, "classifier labels differ"
    train = corpus.role.eq("train")
    y = plain.loc[predicted_rows(plain, corpus), names].copy()
    y.loc[train] = oof.loc[predicted_rows(oof, corpus[train]), names]
    return y.to_numpy(dtype=int)


def predicted_rows(pred: pd.DataFrame, rows: pd.DataFrame) -> pd.Index:
    """``rows``' study ids, all of them with a prediction in ``pred``."""
    missing = ~rows.index.isin(pred.index)
    if missing.any():
        raise SystemExit(
            f"build-prompts: {missing.sum()} corpus studies have no predicted labels "
            f"(e.g. {rows.index[missing][0]})"
        )
    return rows.index


def mix_labels(condition: str, predicted: np.ndarray, reference: np.ndarray):
    """A diagnostic's target labels (0/1 per study and label)."""
    if condition == "labrag-drop-fp":
        return predicted & reference
    if condition == "labrag-add-fn":
        return predicted | reference
    raise ValueError(condition)
