"""Stage ``build-prompts``: one chat prompt per target study, for one condition.

Writes ``prompts/<experiment>/<dataset>/<section>/<condition>/<prompt>/``
``prompts.jsonl.gz``: study_id, set, messages (one user message), retrieved (corpus ids,
best first, with image similarity and, for label ranking, Jaccard), labels (null
without), reference (trimmed only).

An experiment fixes, per dataset, the target and corpus roles (EXPERIMENTS): ``pilot``
(CheXpert Plus pilot split), ``val`` (every val study of MIMIC-CXR and ReXGradient,
corpus train only; their thresholds were set on val, so max-F1 labels are in-sample),
``smoke`` (100 of those val patients, one study each), ``test`` (every test study,
corpus train only, as val), ``train`` (every train-role study of CheXpert Plus,
MIMIC-CXR and ReXGradient, corpus train: prompts for fine-tuning the LLM; the
classifier trained on these studies, so predicted labels are in-sample).

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
their misses added. ``labrag-shuffled.<labeler>``: labrag with another target's
predicted labels (each target gets those of one other target, of another patient, a
fixed derangement): the classifier's label distribution with nothing about this study.
``no-context``: no retrieval and no labels, so its prompt must show the image.
Retrieval never returns a study of the target's patient.

The prompt version decides what a label condition does with the labels: a version with
{labels} and {examples} shows them and ranks on them (LaB-RAG); one without {labels}
only ranks the examples on them (filter only: the labels are never shown); one without
{examples} only shows them (labels only: no retrieval).

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
``--severity`` (labrag-oracle): each label shown with its size or severity from the
``severity`` stage, "Pleural Effusion (small)", for the target and the examples; ranking
unchanged; directory suffix ``.severity``. ``--devices`` (labrag-oracle): the device
types (labels stage, labeler ``devices``) join the label set, for the target and the
examples and in the ranking: "Support Devices, Endotracheal Tube, Enteric Tube";
directory suffix ``.devices``. labrag ``--devices``: the same, the target's device types
predicted by a ``chexbert14.devices`` fine-tune (``--predictions``); without
``--devices`` its predicted device types are left out (CheXbert-14 labels only).
``--incidentals`` (labrag-oracle): the incidental findings (labels stage, labeler
``incidentals``: degenerative spine, aortic calcification, ...) join the label set the
same way, after any device types; directory suffix ``.incidentals``. labrag
``--incidentals``: the same, predicted by a ``chexbert14.devices.incidentals`` fine-tune,
which predicts device types too (shown only with ``--devices``); without
``--incidentals`` its predicted incidental findings are left out.
``--confidence`` (labrag): each target label shown with the classifier's probability,
"Atelectasis (28%)" (its probs file); example labels, label set and ranking unchanged;
directory suffix ``.confidence``.
Multiclass predictions (``--predictions .../chexbert14.mc``, finetune --multiclass)
give CheXbert's four values per target label; their positives rank as usual. A prompt
version with ``label_categories`` (e.g. [Positive, Negative, Uncertain, Unmentioned])
shows the labels under those categories, for the target (predicted) and the examples
(their reference values); any other version shows the positives only. A version with
{absent} also lists the target's labels predicted negative (multiclass only; examples
show their positives as usual); for labrag-oracle, its reference negated labels.

Section ``findings+impression`` (JOINT; only with a ``joint-*`` prompt version, which
asks for both headers): the studies with both sections; examples show "Findings: …
Impression: …"; the reference is the two texts joined, as labelled and scored.
Predicted target labels combine the predict artifact's findings and impression readouts
the way the labels stage combines the reference labels (labels.combine), so predicted,
example and oracle labels share one definition.

``--same reader|site|style`` (any condition with examples): examples only from corpus
studies with the target's reader (the radiologist on the report) or site (the
institution), for a dataset that gives that group, or report-style cluster (the
``styles`` artifact; the target's comes from its own reference report, so this is a
style oracle), ranking otherwise unchanged; directory suffix ``.same-<group>``.

A prompt version with {image} shows the study's preferred image (the one embedded; its
path relative to the CXR root in the record's ``image``). A version with {prior} shows the report of the target patient's previous study
(``prior_text``; its study id in the record's ``prior``), "None" if none. A version with {indication}
shows the target's indication (the adapter's ``indication``; also in the record), any
condition; if its example template has {indication} too, each example shows its own.
"""

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from labrag import artifacts, paths, prompts, retrieval
from labrag.data import (
    DATASETS,
    GROUPS,
    JOIN,
    JOINT,
    JOINT_PARTS,
    add_joint,
    check_section,
    group_files,
    groups,
    has_group,
    has_times,
    priors,
)
from labrag.labelers import devices as device_types
from labrag.labelers import incidentals as incidental_types
from labrag.labelers import radgraph_sides
from labrag.stages import severity as severity_stage
from labrag.stages import styles as styles_stage
from labrag.stages import threshold
from labrag.stages.classify import BINARIZE, LABELER_SUFFIX, MULTICLASS_SUFFIX, POSITIVE
from labrag.stages.labels import (
    COMBINE_RULE,
    DEVICES,
    INCIDENTALS,
    WITH_DEVICES,
    WITH_INCIDENTALS,
    combine,
)

# experiment -> dataset -> target roles, corpus roles and, optionally, a sample of the
# targets: that many patients drawn with the seed, one study each
VAL = {
    "targets": ("val",),  # test stays unseen until the generation config is frozen
    "corpus": ("train",),
}
SMOKE = VAL | {"sample": {"patients": 100, "seed": 20261119}}
TEST = {"targets": ("test",), "corpus": ("train",)}
# LLM fine-tuning prompts: the train studies, retrieving from the other train patients
TRAIN = {"targets": ("train",), "corpus": ("train",)}
EXPERIMENTS = {
    "pilot": {
        "chexpertplus": {
            "targets": ("pilot", "prompt_tuning"),
            "corpus": ("train", "val"),
        },
    },
    "smoke": {"mimic": SMOKE, "rexgradient": SMOKE},
    "val": {"mimic": VAL, "rexgradient": VAL},
    "test": {"mimic": TEST, "rexgradient": TEST},
    "train": {"chexpertplus": TRAIN, "mimic": TRAIN, "rexgradient": TRAIN},
}
# label diagnostics -> how the target's labels mix predicted and reference
DIAGNOSTICS = {
    "labrag-drop-fp": "predicted and reference",
    "labrag-add-fn": "predicted or reference",
}
SHUFFLED = "labrag-shuffled"
SHUFFLE_SEED = 20261126
CONDITIONS = (
    "no-context",
    "retrieval-only",
    "labrag",
    "labrag-oracle",
    SHUFFLED,
    *DIAGNOSTICS,
)
DEFAULT_PROMPT = {
    "no-context": "section-image",
    "retrieval-only": "preprint-naive",
    "labrag": "preprint-simple",
    "labrag-oracle": "preprint-simple",
    SHUFFLED: "preprint-simple",
} | {c: "preprint-simple" for c in DIAGNOSTICS}
K = 5
STYLE = "style"  # report-style clusters (the styles stage), not an adapter group
SAME = (*GROUPS, STYLE)
JOINT_PROMPTS = "joint-"  # prompt versions for JOINT, and only those
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
    severity: bool = False,
    devices: bool = False,
    incidentals: bool = False,
    confidence: bool = False,
    same: str | None = None,
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
    check_section("build-prompts", dataset, section, joint=True)
    joint = section == JOINT
    if joint != prompt.startswith(JOINT_PROMPTS):
        raise SystemExit(
            f"build-prompts: {JOINT} takes the {JOINT_PROMPTS}* prompt versions, "
            "and only it does"
        )
    if joint and (sides or confidence):
        raise SystemExit(f"build-prompts: no --sides or --confidence for {JOINT}")
    version = prompts.load(prompt)
    with_labels = condition.startswith("labrag")  # ranked on and/or shown
    shows_labels = prompts.uses_labels(version)
    with_examples = prompts.uses_examples(version)
    with_image = prompts.uses_image(version)
    oracle = condition == "labrag-oracle"
    predicted = condition in ("labrag", SHUFFLED) or condition in DIAGNOSTICS
    if with_labels:
        fits = shows_labels or with_examples
    else:
        fits = not shows_labels and with_examples == (condition == "retrieval-only")
    if not fits:
        raise SystemExit(f"prompt {prompt!r} does not fit condition {condition!r}")
    if (classifier is not None or predictions) and not predicted:
        raise SystemExit(
            "build-prompts: --classifier/--predictions are for predicted labels"
        )
    if corpus_labels != "reference" and (condition != "labrag" or not with_examples):
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
    if ranking != "labels" and not (with_labels and with_examples):
        raise SystemExit(
            "build-prompts: --ranking is for conditions with labels and examples"
        )
    if sides and condition != "labrag-oracle":
        raise SystemExit("build-prompts: --sides is for labrag-oracle")
    if severity and condition != "labrag-oracle":
        raise SystemExit("build-prompts: --severity is for labrag-oracle")
    if devices and condition not in ("labrag-oracle", "labrag", SHUFFLED):
        raise SystemExit(
            "build-prompts: --devices is for labrag-oracle, labrag and labrag-shuffled"
        )
    if incidentals and condition not in ("labrag-oracle", "labrag", SHUFFLED):
        raise SystemExit(
            "build-prompts: --incidentals is for labrag-oracle, labrag and "
            "labrag-shuffled"
        )
    if sides and severity:
        raise SystemExit("build-prompts: --sides or --severity, not both")
    if confidence and condition != "labrag":
        raise SystemExit("build-prompts: --confidence is for labrag")
    if same is not None and same not in SAME:
        raise SystemExit(f"build-prompts: --same is one of {SAME}")
    if same and not with_examples:
        raise SystemExit(
            f"build-prompts: --same {same} is for conditions with examples"
        )
    if same in GROUPS and not has_group(dataset, same):
        raise SystemExit(f"build-prompts: {dataset} has no {same}s")
    transfer = predictions is not None
    categories = prompts.uses_categories(version)
    absent = prompts.uses_absent(version)
    indication = prompts.uses_indication(version)
    example_indications = prompts.uses_example_indications(version)
    prior = prompts.uses_prior(version)
    if prior and not has_times(dataset):
        raise SystemExit(f"build-prompts: {dataset}'s adapter gives no study times")

    labeler = LABELER + LABELER_SUFFIX[uncertain]
    multiclass = with_types = with_incidentals = False
    if transfer:
        source, clf_name, clf_sections, labeler = parse_classifier(predictions)
        readouts = clf_sections.split(JOIN)
        if joint:
            if not set(JOINT_PARTS) <= set(readouts):
                raise SystemExit(
                    f"build-prompts: {predictions} has no {JOINT} readouts"
                )
            readouts = list(JOINT_PARTS)
        elif section in readouts or len(readouts) == 1:
            readouts = [section if section in readouts else readouts[0]]
        else:
            raise SystemExit(f"build-prompts: {predictions} has no {section} readout")
        readout = readouts[0]
        multiclass = labeler.endswith(MULTICLASS_SUFFIX)
        base = labeler.removesuffix(MULTICLASS_SUFFIX)
        with_incidentals = base == WITH_INCIDENTALS
        with_types = with_incidentals or base == WITH_DEVICES
        if multiclass and (
            condition not in ("labrag", SHUFFLED)
            or confidence
            or corpus_labels != "reference"
        ):
            raise SystemExit(
                "build-prompts: multiclass predictions are for labrag with reference "
                "corpus labels"
            )
        if (
            corpus_labels == "reference"
            and not with_types
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
    if devices and condition in ("labrag", SHUFFLED) and not (transfer and with_types):
        raise SystemExit(
            f"build-prompts: labrag --devices needs --predictions of a {WITH_DEVICES} "
            f"or {WITH_INCIDENTALS} fine-tune"
        )
    if (
        incidentals
        and condition in ("labrag", SHUFFLED)
        and not (transfer and with_incidentals)
    ):
        raise SystemExit(
            "build-prompts: labrag --incidentals needs --predictions of a "
            f"{WITH_INCIDENTALS} fine-tune"
        )
    if (
        transfer
        and with_types
        and (corpus_labels != "reference" or uncertain != "zeros")
    ):
        raise SystemExit(
            f"build-prompts: {labeler} predictions take reference corpus labels and "
            "--uncertain zeros"
        )
    if categories and not multiclass:
        raise SystemExit(
            f"build-prompts: prompt {prompt!r} shows label categories: it needs "
            "multiclass --predictions (chexbert14.mc)"
        )
    if absent and not (multiclass or oracle):
        raise SystemExit(
            f"build-prompts: prompt {prompt!r} lists absent labels: it needs "
            "multiclass --predictions (chexbert14.mc) or labrag-oracle"
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
        if joint:
            config["example_text"] = "Findings: <findings> Impression: <impression>"
    if indication:
        shown_for = "target and examples" if example_indications else "target only"
        config["indication"] = (
            f"{shown_for}: the adapter's indication, {prompts.NO_INDICATION!r} if none"
        )
    if prior:
        config["prior"] = (
            "the report (its sections, the ones it has) of the patient's latest study "
            "with one strictly before the target (adapter study times), "
            f"{prompts.NO_PRIOR!r} if none"
        )
    if with_image:
        image_root = cxr_root / adapter.IMAGE_ROOT
        inputs["images"] = {"root": str(image_root)}  # too many to hash
        config["image"] = "preferred per study (the embedded one)"
    if same == STYLE:
        styles_dir = paths.SHARED_ROOT / styles_stage.relpath(dataset, section)
        inputs["styles"] = artifacts.artifact_record(styles_dir)
        config["same_style"] = (
            "examples only from studies in the target's report-style cluster (the "
            "target's from its own reference report: a style oracle)"
        )
    elif same:
        inputs[f"{same}s"] = [
            artifacts.file_record(cxr_root / f) for f in group_files(dataset, same)
        ]
        config[f"same_{same}"] = f"examples only from studies with the target's {same}"
    if sides:
        sides_dir = paths.SHARED_ROOT / "sides" / dataset / section / LABELER
        inputs["sides"] = artifacts.artifact_record(sides_dir)
        if labeler != LABELER:
            raise SystemExit(f"build-prompts: sides are for {LABELER} labels")
    if severity:
        severity_dir = paths.SHARED_ROOT / severity_stage.relpath(dataset, section)
        inputs["severity"] = artifacts.artifact_record(severity_dir)
        if labeler != LABELER:
            raise SystemExit(f"build-prompts: severity is for {LABELER} labels")
    if devices:
        devices_dir = paths.SHARED_ROOT / "labels" / dataset / section / DEVICES
        inputs["device_labels"] = artifacts.artifact_record(devices_dir)
    if incidentals:
        incidentals_dir = paths.SHARED_ROOT / "labels" / dataset / section / INCIDENTALS
        inputs["incidental_labels"] = artifacts.artifact_record(incidentals_dir)
    if transfer:
        pred_dir = paths.SHARED_ROOT / "predictions" / dataset / predictions.strip("/")
        inputs["predictions"] = artifacts.artifact_record(pred_dir)
        config["predictions"] = predictions
        if joint:
            config["readouts"] = {
                "combined": list(JOINT_PARTS),
                "rule": COMBINE_RULE,
            }
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
                "reference"
                if oracle
                else (
                    "predicted, of another target of another patient (a derangement, "
                    f"seed {SHUFFLE_SEED})"
                    if condition == SHUFFLED
                    else DIAGNOSTICS.get(condition, "predicted")
                )
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
                {
                    "absent": (
                        "target: reference labels negated"
                        if oracle
                        else "target: labels predicted negative"
                    )
                    + "; examples: none"
                }
                if absent
                else {}
            ),
        }
        if not shows_labels:
            config["labels_shown"] = "none: they only rank the examples (filter only)"
        if not with_examples:
            config["ranking"] = "none: no examples (labels only)"
        condition_dir = f"{condition}.{labeler}"
        if predicted:
            short = clf_name.removeprefix(f"{ENCODER}.")
            short = f"{source}.{short}" if transfer else short
            condition_dir = f"{condition}.{short}.{clf_sections}.{labeler}"
        if corpus_labels == "predicted":
            condition_dir += ".corpus-predicted"
        if ranking == "image":
            condition_dir += ".image-ranked"
        if devices:
            config["devices"] = (
                f"the {DEVICES} labels join the label set: target, examples, ranking"
            )
            condition_dir += ".devices"
        elif transfer and with_types:
            config["devices"] = "the predicted device types left out"
        if incidentals:
            config["incidentals"] = (
                f"the {INCIDENTALS} labels join the label set: target, examples, ranking"
            )
            condition_dir += ".incidentals"
        elif transfer and with_incidentals:
            config["incidentals"] = "the predicted incidental findings left out"
        if sides:
            config["sides"] = "reference, target and examples: label (side)"
            condition_dir += ".sides"
        if severity:
            config["severity"] = "reference, target and examples: label (severity)"
            condition_dir += ".severity"
        if confidence:
            config["confidence"] = "target labels: label (probability, % rounded, 1-99)"
            condition_dir += ".confidence"
    else:
        if with_examples:
            config["ranking"] = "cosine"
        condition_dir = condition
    if same:
        condition_dir += f".same-{same}"

    prior_texts = {}  # target study_id -> (prior study_id, prior report text)

    def build(out: Path) -> dict:
        labels = artifacts.read_table(lab_dir / "labels.csv").set_index("study_id")
        extra = [devices_dir] if devices else []
        extra += [incidentals_dir] if incidentals else []
        for d in extra:  # rule label sets, joined after CheXbert-14's
            typed = artifacts.read_table(d / "labels.csv").set_index("study_id")
            assert labels.index.isin(
                typed.index
            ).all(), f"a study has no {d.name} labels"
            labels = labels.join(typed)
        names = list(labels.columns)
        if transfer:
            tables = [
                artifacts.read_table(pred_dir / f"predictions.{r}.csv")
                for r in readouts
            ]
            transferred = (
                combine(tables, multiclass)
                if joint
                else tables[0].set_index("study_id")
            )
            drop = [
                *(device_types.NAMES if with_types and not devices else []),
                *(
                    incidental_types.NAMES
                    if with_incidentals and not incidentals
                    else []
                ),
            ]
            transferred = transferred.drop(columns=drop)
            names = list(transferred.columns)
        studies = adapter.load_studies(cxr_root).set_index("study_id")
        if joint:
            studies = add_joint(studies)
        if indication and "indication" not in studies:
            raise SystemExit(f"build-prompts: no indication in {dataset}'s adapter")
        studies["role"] = adapter.classifier_roles(studies, paths.SPLITS_DIR)
        if prior:
            prior_texts.update(prior_text(studies, adapter, cxr_root))
        studies = studies.loc[labels.index]  # the studies with this section
        target = studies[studies.role.isin(exp["targets"])]
        if "sample" in exp:
            target = sample_targets(target, **exp["sample"])
        corpus = studies[studies.role.isin(exp["corpus"])]
        if with_examples:
            embedded = artifacts.read_table(emb_dir / "studies.csv")
            embedded = embedded.set_index("study_id")
            embedded["row"] = np.arange(len(embedded))
            emb = np.load(emb_dir / "embeddings.npy")

            def embeddings(rows: pd.DataFrame) -> np.ndarray:
                # The embedded image must be the one the adapter picks for the study.
                e = embedded.loc[rows.index]
                assert (e.image_path == rows.image_path).all(), "image mismatch"
                return emb[e.row.to_numpy()]

            target_emb, corpus_emb = embeddings(target), embeddings(corpus)
            t_subject = target.subject_id.to_numpy()
            c_subject = corpus.subject_id.to_numpy()
            if same == STYLE:
                table = artifacts.read_table(styles_dir / "styles.csv")
                group = table.set_index("study_id")["style"]
            elif same:
                group = groups(dataset, same, cxr_root)
            if same:
                t_group = group.reindex(target.index)
                assert t_group.notna().all(), f"a target study has no {same}"
                t_group = t_group.to_numpy()
                c_group = group.reindex(corpus.index).to_numpy()

            def exclude(rows: slice) -> np.ndarray:
                mask = t_subject[rows, None] == c_subject[None, :]
                if same:
                    mask |= t_group[rows, None] != c_group[None, :]
                return mask

        target_y = corpus_y = None
        if with_labels:

            def reference(rows: pd.DataFrame) -> np.ndarray:
                y = labels.loc[rows.index, names].isin(POSITIVE[uncertain])
                return y.to_numpy(dtype=int)

            if corpus_labels == "reference":
                assert names == list(labels.columns), "classifier labels differ"
            if not with_examples:
                corpus_y = None
            elif transfer and corpus_labels == "predicted":
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
                if absent:
                    target_codes = labels.loc[target.index, names].to_numpy()
            else:
                pred = artifacts.read_table(clf_dir / f"predictions.{section}.csv")
                pred = pred.set_index("study_id")
                assert list(pred.columns) == names, "classifier labels differ"
                target_y = pred.loc[target.index, names].to_numpy(dtype=int)
            if condition in DIAGNOSTICS:
                target_y = mix_labels(condition, target_y, reference(target))
            if condition == SHUFFLED:
                donor = donors(target.subject_id.to_numpy(), SHUFFLE_SEED)
                target_y = target_y[donor]
                if target_codes is not None:
                    target_codes = target_codes[donor]
        if with_examples:
            top, top_sim, top_jaccard = retrieval.retrieve(
                K,
                target_emb,
                corpus_emb,
                exclude,
                target_y,
                corpus_y,
                by_labels=ranking == "labels",
            )
        else:
            top = np.zeros((len(target), 0), dtype=int)

        detail = {}  # (study_id, label) -> side or severity, shown after the label
        if sides:
            table = artifacts.read_table(sides_dir / "sides.csv")
            detail = dict(zip(zip(table.study_id, table.label), table.side))
        if severity:
            table = artifacts.read_table(severity_dir / "severity.csv")
            detail = dict(zip(zip(table.study_id, table.label), table.severity))

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
            return [radgraph_sides.sided(n, detail.get((study_id, n))) for n in shown]

        reports = (
            ("Findings: " + corpus.findings + " Impression: " + corpus.impression)
            if joint
            else corpus[section]
        ).to_numpy()
        if example_indications:
            corpus_indications = corpus.indication.fillna("").to_numpy()
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
            retrieved = None
            if with_examples:
                retrieved = {
                    "retrieved": corpus_ids[idx].tolist(),
                    "retrieved_similarity": top_sim[i].tolist(),
                }
            if with_examples and with_labels:
                retrieved["retrieved_jaccard"] = top_jaccard[i].tolist()
            if categories:
                labels_i = prompts.categories(target_codes[i], names, shown)
            else:
                labels_i = (
                    positives(target_y[i], study_id, True) if shows_labels else None
                )
            absent_i = (
                [n for n, c in zip(names, target_codes[i]) if c == 0]
                if absent
                else None
            )
            shown_indications = (
                [corpus_indications[j] for j in idx] if example_indications else None
            )
            records.append(
                record(
                    study_id,
                    row,
                    examples,
                    labels_i,
                    retrieved,
                    absent_i,
                    shown_indications,
                )
            )
        summary = write(out, records)
        if with_examples:
            summary["n_corpus"] = len(corpus)
        if with_examples and with_labels:
            summary["top1_jaccard_1"] = float((top_jaccard[:, 0] == 1).mean())
            summary["mean_retrieved_jaccard"] = float(top_jaccard.mean())
        print(json.dumps(summary, indent=2))
        return summary

    def record(
        study_id,
        row,
        examples=(),
        labels_i=None,
        retrieved=None,
        absent_i=None,
        example_indications_i=None,
    ) -> dict:
        indication_i = None
        if indication:
            indication_i = "" if pd.isna(row.indication) else row.indication
        prior_id, prior_i = (
            prior_texts.get(study_id, (None, "")) if prior else (None, None)
        )
        r = {
            "study_id": study_id,
            "set": row.role,
            "messages": prompts.render(
                version,
                list(examples),
                labels_i,
                section,
                absent_i,
                indication_i,
                example_indications_i,
                prior_i,
            ),
        }
        if with_image:
            r["image"] = f"{adapter.IMAGE_ROOT}/{row.image_path}"
        shown = {"labels": labels_i} | ({"absent": absent_i} if absent else {})
        if indication:
            shown["indication"] = indication_i or None
        if prior:
            shown["prior"] = prior_id
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


def prior_text(studies: pd.DataFrame, adapter, cxr_root: Path) -> dict:
    """study_id -> (prior study_id, its report: "Findings: ... Impression: ...", the
    sections it has) for each study (index study_id) with a prior: the patient's latest
    earlier study with any section (data.priors)."""
    parts = list(adapter.SECTIONS)
    reported = studies[studies[parts].notna().any(axis=1)]
    prior_of = priors(reported, adapter.load_study_times(cxr_root)).dropna()
    rows = reported.loc[prior_of.to_numpy(), parts]
    texts = [
        " ".join(
            f"{prompts.SECTION_NAMES[p]}: {t}" for p, t in r.items() if pd.notna(t)
        )
        for _, r in rows.iterrows()
    ]
    return dict(zip(prior_of.index, zip(prior_of.to_numpy(), texts)))


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


def donors(subjects: np.ndarray, seed: int) -> np.ndarray:
    """Per target, the index of the target whose labels it gets (labrag-shuffled): a
    random cyclic order, each target taking the next one's; shifted further until no
    target gets a study of its own patient."""
    order = np.random.default_rng(seed).permutation(len(subjects))
    for shift in range(1, len(subjects)):
        donor = np.empty(len(subjects), dtype=int)
        donor[order] = np.roll(order, -shift)
        if not (subjects[donor] == subjects).any():
            return donor
    raise SystemExit(
        "build-prompts: every target is of one patient; nothing to shuffle"
    )


def mix_labels(condition: str, predicted: np.ndarray, reference: np.ndarray):
    """A diagnostic's target labels (0/1 per study and label)."""
    if condition == "labrag-drop-fp":
        return predicted & reference
    if condition == "labrag-add-fn":
        return predicted | reference
    raise ValueError(condition)
