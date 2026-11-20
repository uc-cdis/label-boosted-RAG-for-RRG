"""Stage ``finetune``: RAD-DINO fine-tuned end to end with the attention head, on one or more datasets' labels.

Writes ``classifiers/<datasets>/raddino.ft-<classifier>-seed<seed>/<sections>/<labeler>[.u1]/``
(datasets and sections joined by ``-``, e.g. ``chexpertplus-mimic-rexgradient`` and
``findings-impression``): model.pt (backbone + head state dict, labels, sections,
datasets, pooled-val thresholds), history.csv (one row per epoch), metrics.csv (per
dataset, section and label on val, at the pooled-val thresholds).

Its studies: each dataset's studies with labels in any of the sections. Trained on
every dataset's ``train`` role together; the epoch is picked on their ``val`` roles,
each dataset counting equally (classifiers/finetune.py). Thresholds for use come from
the target dataset's own val in ``predict``; the pooled-val max-F1 thresholds stored
here are the fallback for a target without labels. Classifiers: ``attn`` (the head on
CLS + patch tokens) and ``attn-cls`` (CLS only, the control).

Only report-labelled datasets (CheXbert-14 on report text): every dataset's labels
are then the same labeler on the same section.

``--multiclass``: each label is one of CheXbert's four values (positive, negative,
uncertain, blank) instead of positive vs the rest, softmax cross-entropy; the predicted
value is the most probable class, so there are no thresholds. Labeler directory
``chexbert14.mc``.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from labrag import artifacts, paths, weights
from labrag.classifiers import attention, finetune, logistic
from labrag.data import DATASETS, JOIN, labels_only, split_sections
from labrag.stages import classify

CLASSIFIERS = ("attn", "attn-cls")
ENCODER = "raddino"
MAX_WORKERS = 128  # decoding + augmentation is the bottleneck; split over the GPUs


def split_datasets(stage: str, datasets: str) -> list[str]:
    """The datasets of a joined name, each once, in DATASETS order."""
    parts = datasets.split(JOIN)
    order = list(DATASETS)
    unknown = [d for d in parts if d not in DATASETS]
    if unknown:
        raise SystemExit(f"{stage}: unknown datasets {unknown}; choose from {order}")
    if parts != sorted(set(parts), key=order.index):
        raise SystemExit(f"{stage}: join datasets once each, in the order {order}")
    return parts


def run(
    dataset: str,
    section: str,
    labeler: str = "chexbert14",
    classifier: str = "attn",
    uncertain: str = "zeros",
    seed: int | None = None,
    multiclass: bool = False,
    overwrite: bool = False,
    cxr_root: Path | None = None,
) -> Path:
    datasets = split_datasets("finetune", dataset)
    if multiclass and uncertain != "zeros":
        raise SystemExit("finetune: --multiclass keeps uncertain as its own class")
    sections = None
    for d in datasets:
        if labels_only(d):
            raise SystemExit(f"finetune: {d} has release labels, not report labels")
        sections = split_sections("finetune", d, section)
    if labeler != "chexbert14":
        raise SystemExit("finetune: report labels are chexbert14")
    if classifier not in CLASSIFIERS:
        raise SystemExit(f"finetune: --classifier {list(CLASSIFIERS)}")
    seed = 0 if seed is None else seed
    cxr_root = cxr_root or paths.CXR_ROOT
    model_dir, weights_record = weights.resolve(embed_weights())
    lab_dirs = {
        (d, s): paths.SHARED_ROOT / "labels" / d / s / labeler
        for d in datasets
        for s in sections
    }
    inputs = {
        "labels": {
            f"{d}/{s}": artifacts.artifact_record(p) for (d, s), p in lab_dirs.items()
        },
        "sources": {
            d: [artifacts.file_record(cxr_root / f) for f in DATASETS[d].SOURCE_FILES]
            for d in datasets
        },
        "splits": {
            d: [
                artifacts.file_record(paths.SPLITS_DIR / f)
                for f in DATASETS[d].SPLIT_FILES
            ]
            for d in datasets
        },
        "images": {d: str(cxr_root / DATASETS[d].IMAGE_ROOT) for d in datasets},
        "weights": weights_record,
    }
    config = {
        "stage": "finetune",
        "datasets": datasets,
        "sections": sections,
        "encoder": ENCODER,
        "labeler": labeler,
        "classifier": classifier,
        "params": finetune.PARAMS,
        "binarize": classify.BINARIZE[uncertain],
        "image": "preferred frontal per study, min-max to 8-bit grayscale",
        "fit": "every dataset's train role; each section's loss on its labelled studies",
        "threshold": "max F1 per section on the pooled val (fallback; predict uses the "
        "target's own val)",
        "consistency": classify.CONSISTENCY,
        "seed": seed,
    }
    if classifier == "attn-cls":
        config["input"] = "CLS token only"
    classes = 1
    suffix = classify.LABELER_SUFFIX[uncertain]
    if multiclass:
        classes, suffix = len(finetune.MULTICLASS), classify.MULTICLASS_SUFFIX
        del config["binarize"]
        config |= {
            "classes": classify.MULTICLASS_CLASSES,
            "loss": "softmax cross-entropy over the classes, each study's labelled "
            "entries",
            "select": "epoch with the best mean over datasets of their mean val AUROC "
            "of the positive class (one vs rest)",
            "threshold": "none: the most probable class",
        }
    name = f"{ENCODER}.ft-{classifier}-seed{seed}"
    relpath = (
        f"classifiers/{JOIN.join(datasets)}/{name}/{JOIN.join(sections)}/"
        f"{labeler}{suffix}"
    )

    def build(out: Path) -> dict:
        rows, names = [], None
        for d in datasets:
            adapter = DATASETS[d]
            labels = {
                s: artifacts.read_table(lab_dirs[d, s] / "labels.csv").set_index(
                    "study_id"
                )
                for s in sections
            }
            for t in labels.values():
                names = names or list(t.columns)
                assert list(t.columns) == names, "labels differ"
            studies = adapter.load_studies(cxr_root).set_index("study_id")
            studies["role"] = adapter.classifier_roles(studies, paths.SPLITS_DIR)
            labelled = sorted(set().union(*(t.index for t in labels.values())))
            missing = set(labelled) - set(studies.index)
            assert (
                not missing
            ), f"{d}: {len(missing)} labelled studies not in the adapter"
            studies = studies.loc[labelled]
            rows.append((d, studies, labels))

        n = len(names)
        ids, group, role, image_paths, blocks = [], [], [], [], []
        Y_parts, mask_parts = [], []
        for d, studies, labels in rows:
            root = cxr_root / DATASETS[d].IMAGE_ROOT
            ids += list(studies.index)
            group += [d] * len(studies)
            role += list(studies.role)
            image_paths += [root / p for p in studies.image_path]
            Y = np.zeros((len(studies), n * len(sections)), dtype=int)
            mask = np.zeros(Y.shape, dtype=bool)
            for i, s in enumerate(sections):
                has = studies.index.isin(labels[s].index)
                codes = labels[s].loc[studies.index[has], names]
                if multiclass:
                    y = finetune.class_index(codes.to_numpy())
                else:
                    y = codes.isin(classify.POSITIVE[uncertain]).to_numpy(dtype=int)
                Y[has, i * n : (i + 1) * n] = y
                mask[has, i * n : (i + 1) * n] = True
            Y_parts.append(Y)
            mask_parts.append(mask)
        Y, mask = np.concatenate(Y_parts), np.concatenate(mask_parts)
        group, role = np.array(group), np.array(role)
        train, val = role == "train", role == "val"

        world = max(1, torch.cuda.device_count())
        cpus = len(os.sched_getaffinity(0))
        workers = min(MAX_WORKERS, max(0, cpus - 1)) // world
        state, P_val, history = finetune.train(
            model_dir,
            image_paths,
            Y,
            mask,
            train,
            val,
            group,
            len(sections),
            seed,
            out,
            cls_only=classifier == "attn-cls",
            world=world,
            workers=workers,
            classes=classes,
        )

        Yv, mv, gv = Y[val], mask[val], group[val]
        P_full = P_val
        if multiclass:
            positive = finetune.MULTICLASS.index("positive")
            P_val = finetune.positive_probs(P_full, len(sections), classes)
            Yv = (Yv == positive).astype(int)
            Y = (Y == positive).astype(int)  # train_pos below counts positives
        threshold, metrics = [], []
        for i, s in enumerate(sections):
            b = slice(i * n, (i + 1) * n)
            lab = mv[:, i * n]  # a section's labels are all present or all absent
            if multiclass:
                pred = classify.codes_positive(
                    classify.argmax_codes(P_full, i, len(sections), names)
                )
            else:
                t = np.array(
                    [
                        logistic.f1_threshold(y, p)
                        for y, p in zip(Yv[lab, b].T, P_val[lab, b].T)
                    ]
                )
                threshold.append(t)
                pred = classify.predict_labels(P_val[:, b], t, names, False)
            for d in datasets:
                r = lab & (gv == d)
                metrics.append(
                    pd.DataFrame(
                        {
                            "dataset": d,
                            "section": s,
                            "label": names,
                            "train_pos": Y[
                                train & (group == d) & mask[:, i * n], b
                            ].sum(0),
                            **({} if multiclass else {"threshold": t}),
                            **classify._scores("val", Yv[r, b], P_val[r, b], pred[r]),
                        }
                    )
                )
        metrics = pd.concat(metrics, ignore_index=True)
        metrics.to_csv(out / "metrics.csv", index=False)
        print(metrics.to_string(index=False))
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        torch.save(
            {
                "labels": names,
                "sections": sections,
                "datasets": datasets,
                "state_dict": state,
                "threshold": None if multiclass else np.stack(threshold),
                "classes": list(finetune.MULTICLASS) if multiclass else None,
            },
            out / "model.pt",
        )
        best = max(history, key=lambda h: h["val_auroc"])
        return {
            "n_studies": len(ids),
            "n_by_dataset_role": pd.Series(list(zip(group, role)))
            .map(lambda x: f"{x[0]}/{x[1]}")
            .value_counts()
            .sort_index()
            .to_dict(),
            "epochs": len(history),
            "best_epoch": best["epoch"],
            "val_mean_auroc": best["val_auroc"],
            "val_mean_auroc_by_dataset": {
                k.split(".", 1)[1]: v
                for k, v in best.items()
                if k.startswith("val_auroc.")
            },
            "gpus": world,
            "workers_per_gpu": workers,
            "device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            ),
        }

    return artifacts.publish(relpath, config, inputs, build, overwrite)


def embed_weights() -> str:
    from labrag.stages import embed

    return embed.ENCODERS[ENCODER]
