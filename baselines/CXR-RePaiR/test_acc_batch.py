import argparse

import numpy as np
import pandas as pd
import scipy.stats as st
import torch
from sklearn.metrics import f1_score

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


def bootstrap_f1(args):
    if args.dataset == "CheXpert":
        true_labels = pd.read_csv(args.chexpert_true_embeddings)[useful_labels]
    else:
        true_labels = pd.read_csv(args.bootstrap_dir + "labels.csv").fillna(0)[
            useful_labels
        ]

    pred_labels = pd.read_csv(args.dir + "labeled_reports.csv").fillna(0)[useful_labels]

    np_true_labels = true_labels.to_numpy()
    np_pred_labels = pred_labels.to_numpy()
    np_pred_labels[np_pred_labels == -1] = 0
    np_true_labels[np_true_labels == -1] = 0
    opts = np.array([0, 1])
    assert np.all(np.isin(np_pred_labels, opts))  # make sure all 0s and 1s

    scores = []
    for i in range(10):  # 10 bootstrap
        indices = np.loadtxt(args.bootstrap_dir + str(i) + "/indices.txt", dtype=int)
        batch_score = f1_batch(indices, np_pred_labels, np_true_labels)
        scores.append(batch_score)
    interval = st.t.interval(
        0.95, df=len(scores) - 1, loc=np.mean(scores), scale=st.sem(scores)
    )
    mean = sum(scores) / len(scores)
    print(f"f1 mean: {round(mean, 3)}, plus or minus {round(mean - interval[0], 3)}")


def f1_batch(indices, pred_labels, true_labels):
    f1_macro = f1_score(
        true_labels[indices, :], pred_labels[indices, :], average="macro"
    )
    return f1_macro


def bootstrap_embeddings_accuracy(args):
    if args.dataset == "CheXpert":
        label_embeds = torch.load(args.chexpert_true_embeddings)
    else:
        label_embeds = torch.load(args.bootstrap_dir + "/embeddings.pt")
    np_label_embeds = torch.stack([*label_embeds.values()], dim=0).numpy()
    pred_embeds = torch.load(args.dir + "embeddings.pt")
    np_pred_embeds = torch.stack([*pred_embeds.values()], dim=0).numpy()
    assert np_label_embeds.shape == np_pred_embeds.shape

    scores = []
    for i in range(10):  # 10 bootstrap
        indices = np.loadtxt(args.bootstrap_dir + str(i) + "/indices.txt", dtype=int)
        batch_score = embeddings_accuracy_batch(
            indices, np_pred_embeds, np_label_embeds
        )
        scores.append(batch_score)
    interval = st.t.interval(
        0.95, df=len(scores) - 1, loc=np.mean(scores), scale=st.sem(scores)
    )
    mean = sum(scores) / len(scores)
    print(f"s_emb mean: {round(mean, 3)}, plus or minus {round(mean - interval[0], 3)}")


def embeddings_accuracy_batch(indices, pred_embeds, true_embeds):
    pred_sample = pred_embeds[indices, :]
    true_sample = true_embeds[indices, :]
    sim_scores = (true_sample * pred_sample).sum(axis=1) / (
        np.linalg.norm(pred_sample, axis=1) * np.linalg.norm(true_sample, axis=1)
    )
    return sim_scores.mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap testing framework for CXR-RePaiR and baseline methods"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MIMIC-CXR",
        choices=["CheXpert", "MIMIC-CXR"],
        help="dataset to evaluate on",
    )
    parser.add_argument(
        "--dir",
        type=str,
        help="directory where labels and embeddings are",
        required=True,
    )
    parser.add_argument(
        "--bootstrap_dir",
        type=str,
        required=True,
        help="directory where boostrap indices and labels are stored",
    )
    parser.add_argument("--include_s_emb", type=bool, required=False, default=False)
    parser.add_argument(
        "--chexpert_true_labels",
        type=str,
        required=False,
        help="path of chexpert true labels",
    )
    parser.add_argument(
        "--chexpert_true_embeddings",
        type=str,
        required=False,
        help="path of chexpert true embeddings",
    )

    args = parser.parse_args()

    separator = "/"
    print(f"metrics on {args.dir.split(separator)[-2]}")
    bootstrap_f1(args)
    if args.include_s_emb:
        bootstrap_embeddings_accuracy(args)
