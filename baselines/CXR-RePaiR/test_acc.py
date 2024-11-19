import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from tabulate import tabulate

cxr_labels = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]
useful_labels = cxr_labels


def main(args):
    print("F1")
    calculate_f1(args.dir, args.dataset, args.gt_labels_path)
    print("Precision Recall:")
    calculate_prec_recall(args.dir, args.dataset, args.gt_labels_path)
    # print("Embeddings Similarity")
    # print(calculate_s_emb(args.dir, args.gt_embeddings_path).mean())


def calculate_f1(dir, dataset, gt_labels_path):
    if dataset == "CheXpert":
        true_labels = pd.read_csv(gt_labels_path)[useful_labels]
    elif dataset == "MIMIC-CXR":
        true_labels = pd.read_csv(gt_labels_path).fillna(0)[useful_labels]

    pred_labels = pd.read_csv(dir + "labeled_reports.csv", index_col=False).fillna(0)[
        useful_labels
    ]

    np_true_labels = true_labels.to_numpy()
    np_pred_labels = pred_labels.to_numpy()
    np_pred_labels[np_pred_labels == -1] = 0
    np_true_labels[np_true_labels == -1] = 0
    opts = np.array([0, 1])
    assert np.all(np.isin(np_pred_labels, opts))

    f1_macro = f1_score(np_true_labels, np_pred_labels, average="macro")
    f1_micro = f1_score(np_true_labels, np_pred_labels, average="micro")
    print("F1 Macro score:", f1_macro)
    print("F1 Micro score:", f1_micro)


def calculate_prec_recall(dir, dataset, gt_labels_path):
    if dataset == "CheXpert":
        true_labels = pd.read_csv(gt_labels_path)[useful_labels]
    elif dataset == "MIMIC-CXR":
        true_labels = pd.read_csv(gt_labels_path).fillna(0)[useful_labels]

    pred_labels = pd.read_csv(dir + "labeled_reports.csv", index_col=False).fillna(0)[
        useful_labels
    ]

    np_true_labels = true_labels.to_numpy()
    np_pred_labels = pred_labels.to_numpy()
    np_true_labels[np_true_labels == -1] = 0
    np_pred_labels[np_pred_labels == -1] = 0

    precs = []
    recalls = []
    for i in range(len(useful_labels)):
        y_true = np_true_labels[:, i]
        y_pred = np_pred_labels[:, i]
        opts = np.array([0, 1])
        if not np.all(np.isin(y_true, opts)):
            print(np.unique(y_true))

        assert np.all(np.isin(y_true, opts))
        assert np.all(np.isin(y_pred, opts))
        prec, recall, _, _ = precision_recall_fscore_support(
            y_true, y_pred, zero_division=1, average="binary"
        )
        precs.append(prec)
        recalls.append(recall)
    precs.append(np.array(precs).mean())
    recalls.append(np.array(recalls).mean())

    _df = pd.DataFrame([precs], columns=[*useful_labels, "Average"])
    print("Precision:")
    print(tabulate(_df, headers="keys", tablefmt="psql", showindex=False))
    _df = pd.DataFrame([recalls], columns=[*useful_labels, "Average"])
    print("Recall:")
    print(tabulate(_df, headers="keys", tablefmt="psql", showindex=False))


def calculate_s_emb(dir, gt_embeddings_path):
    label_embeds = torch.load(gt_embeddings_path)
    np_label_embeds = torch.stack([*label_embeds.values()], dim=0).numpy()
    np_label_embeds = np_label_embeds

    pred_embeds = torch.load(dir + "embeddings.pt")
    np_pred_embeds = torch.stack([*pred_embeds.values()], dim=0).numpy()
    assert np_label_embeds.shape == np_pred_embeds.shape
    # calc cosine sim
    sim_scores = (np_label_embeds * np_pred_embeds).sum(axis=1) / (
        np.linalg.norm(np_pred_embeds, axis=1) * np.linalg.norm(np_label_embeds, axis=1)
    )
    assert len(sim_scores) == np_label_embeds.shape[0]
    return sim_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Testing framework for CXR-RePaiR and baseline methods"
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="directory where predicted labels and embeddings are",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default="MIMIC-CXR",
        choices=["CheXpert", "MIMIC-CXR"],
        help="dataset to evaluate on",
    )
    parser.add_argument(
        "--gt_labels_path",
        type=str,
        required=True,
        help="path to where gt labels are stored",
    )
    parser.add_argument(
        "--gt_embeddings_path",
        type=str,
        required=False,
        help="path to gt CheXbert embeddings for s_emb score",
    )
    args = parser.parse_args()

    main(args)
