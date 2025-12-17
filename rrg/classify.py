import argparse
import os
import pickle
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _data import (
    DEFAULT_DICOM_ID_COL,
    DEFAULT_IMG_EMBED_KEY,
    DEFAULT_LABELS,
    DEFAULT_PATIENT_ID_COL,
    DEFAULT_SPLIT_COL,
    DEFAULT_STUDY_ID_COL,
    DEFAULT_VIEW_COL,
    DEFAULT_VIEW_ORDER,
    get_per_study_data,
    get_split_features,
    get_split_samples,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from tqdm import tqdm

DEFAULT_MAX_ITER = 500
DEFAULT_NUM_WORKERS = -1


def train_image_classifier(
    *,  # enforce kwargs
    split_csv: str,
    metadata_csv: str,
    label_csv: str,
    feature_h5: str,
    output_results: str,
    patient_id_col: str = DEFAULT_PATIENT_ID_COL,
    study_id_col: str = DEFAULT_STUDY_ID_COL,
    dicom_id_col: str = DEFAULT_DICOM_ID_COL,
    split_col: str = DEFAULT_SPLIT_COL,
    view_col: str = DEFAULT_VIEW_COL,
    labels: list[str] = DEFAULT_LABELS,
    view_order: list[str] = DEFAULT_VIEW_ORDER,
    feature_key: str = DEFAULT_IMG_EMBED_KEY,
    max_iter: int = DEFAULT_MAX_ITER,
    num_workers: int = DEFAULT_NUM_WORKERS,
):
    assert not os.path.exists(output_results)
    os.makedirs(output_results)

    sample_df = get_per_study_data(
        split_csv=split_csv,
        metadata_csv=metadata_csv,
        label_csv=label_csv,
        patient_id_col=patient_id_col,
        study_id_col=study_id_col,
        dicom_id_col=dicom_id_col,
        split_col=split_col,
        view_col=view_col,
        labels=labels,
        view_order=view_order,
    )
    split_features = get_split_features(
        feature_h5=feature_h5,
        feature_key=feature_key,
        sample_df=sample_df,
        patient_id_col=patient_id_col,
        study_id_col=study_id_col,
        dicom_id_col=dicom_id_col,
        split_col=split_col,
    )
    split_labels = get_split_samples(
        sample_df=sample_df,
        split_col=split_col,
    )

    X_train, y_train = split_features["train"], split_labels["train"]
    X_val, y_val = split_features["validate"], split_labels["validate"]
    X_test, y_test = split_features["test"], split_labels["test"]

    # TODO hardcoded positive value
    y_train[labels] = (y_train[labels] == 1).astype(int)
    y_val[labels] = (y_val[labels] == 1).astype(int)
    y_test[labels] = (y_test[labels] == 1).astype(int)

    y_prob_train = y_train.copy()
    y_prob_val = y_val.copy()
    y_prob_test = y_test.copy()

    y_pred_roc_train = y_train.copy()
    y_pred_roc_val = y_val.copy()
    y_pred_roc_test = y_test.copy()

    y_pred_pr_train = y_train.copy()
    y_pred_pr_val = y_val.copy()
    y_pred_pr_test = y_test.copy()

    models = dict()
    for label in tqdm(labels, desc="Training Label"):
        # TODO hardcoded model type
        model = LogisticRegression(
            penalty="l2",
            solver="saga",
            max_iter=max_iter,
            n_jobs=num_workers,
        )
        model.fit(X_train, y_train[label].to_numpy())
        y_prob_train[label] = model.predict_proba(X_train)[:, 1]
        y_prob_val[label] = model.predict_proba(X_val)[:, 1]
        y_prob_test[label] = model.predict_proba(X_test)[:, 1]

        roc_threshold = get_optimal_threshold(
            trues=y_val[label].to_numpy(),
            probs=y_prob_val[label].to_numpy(),
            method="roc",
        )
        y_pred_roc_train[label] = (y_prob_train[label] > roc_threshold).astype(int)
        y_pred_roc_val[label] = (y_prob_val[label] > roc_threshold).astype(int)
        y_pred_roc_test[label] = (y_prob_test[label] > roc_threshold).astype(int)

        pr_threshold = get_optimal_threshold(
            trues=y_val[label].to_numpy(),
            probs=y_prob_val[label].to_numpy(),
            method="pr",
        )
        y_pred_pr_train[label] = (y_prob_train[label] > pr_threshold).astype(int)
        y_pred_pr_val[label] = (y_prob_val[label] > pr_threshold).astype(int)
        y_pred_pr_test[label] = (y_prob_test[label] > pr_threshold).astype(int)

        models[label] = model

    # save models and predictions to disk
    with open(os.path.join(output_results, "models.pkl"), "wb") as f:
        pickle.dump(models, f)

    y_train.to_csv(os.path.join(output_results, "train_true.csv"), index=False)
    y_val.to_csv(os.path.join(output_results, "val_true.csv"), index=False)
    y_test.to_csv(os.path.join(output_results, "test_true.csv"), index=False)

    y_prob_train.to_csv(os.path.join(output_results, "train_prob.csv"), index=False)
    y_prob_val.to_csv(os.path.join(output_results, "val_prob.csv"), index=False)
    y_prob_test.to_csv(os.path.join(output_results, "test_prob.csv"), index=False)

    y_pred_roc = merge_predictions_as_reference(
        reference_label_csv=label_csv,
        train_pred=y_pred_roc_train,
        val_pred=y_pred_roc_val,
        test_pred=y_pred_roc_test,
        study_id_col=study_id_col,
        labels=labels,
    )
    y_pred_roc.to_csv(os.path.join(output_results, "pred_roc.csv"), index=False)
    y_pred_pr = merge_predictions_as_reference(
        reference_label_csv=label_csv,
        train_pred=y_pred_pr_train,
        val_pred=y_pred_pr_val,
        test_pred=y_pred_pr_test,
        study_id_col=study_id_col,
        labels=labels,
    )
    y_pred_pr.to_csv(os.path.join(output_results, "pred_pr.csv"), index=False)

    # save plots
    for df_trues, df_probs, split in [
        (y_train, y_prob_train, "train"),
        (y_val, y_prob_val, "val"),
        (y_test, y_prob_test, "test"),
    ]:
        plot_roc_curve(
            df_trues=df_trues,
            df_probs=df_probs,
            labels=labels,
            title=f"{split.title()} ROC Curve",
            output_path=os.path.join(output_results, f"{split}_curve_roc.png"),
        )
        plot_pr_curve(
            df_trues=df_trues,
            df_probs=df_probs,
            labels=labels,
            title=f"{split.title()} PR Curve",
            output_path=os.path.join(output_results, f"{split}_curve_pr.png"),
        )


def get_optimal_threshold(
    *,  # enforce kwargs
    trues: np.ndarray,
    probs: np.ndarray,
    method: Literal["roc", "pr"] = "roc",
) -> float:
    if method == "pr":
        precision, recall, thresholds = precision_recall_curve(trues, probs)
        scores = 2 * (precision * recall) / (precision + recall)  # F1
    elif method == "roc":
        fpr, tpr, thresholds = roc_curve(trues, probs)
        scores = tpr - fpr  # Youden's J
    else:
        raise ValueError(f"Unknown method: {method}")
    idx = np.argmax(scores)
    threshold = thresholds[idx]
    return threshold


def merge_predictions_as_reference(
    *,  # enforce kwargs
    reference_label_csv: str,
    train_pred: pd.DataFrame,
    val_pred: pd.DataFrame,
    test_pred: pd.DataFrame,
    study_id_col: str,
    labels: list[str],
) -> pd.DataFrame:
    ref = pd.read_csv(reference_label_csv)
    ref_cols = [col for col in ref.columns if col not in labels]
    ref = ref[ref_cols]
    pred = pd.concat([train_pred, val_pred, test_pred])[[study_id_col] + labels]
    return ref.merge(pred, on=study_id_col)


def plot_roc_curve(
    *,  # enforce kwargs
    df_trues: pd.DataFrame,
    df_probs: pd.DataFrame,
    labels: list[str],
    title: str,
    output_path: str,
):
    colors = plt.colormaps["tab20"]
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 9
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.plot([0, 1], [0, 1], "--", color="red", alpha=0.5)
    for i, label in enumerate(labels):
        color = colors(i)
        fpr, tpr, _ = roc_curve(df_trues[label], df_probs[label])
        auroc = auc(fpr, tpr)
        if label == "Enlarged Cardiomediastinum":
            label = "Enl. Card."
        ax.plot(fpr, tpr, label=f"{label} (AUC = {auroc:.3f})", color=color)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right", fontsize="xx-small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=1000)


def plot_pr_curve(
    *,  # enforce kwargs
    df_trues: pd.DataFrame,
    df_probs: pd.DataFrame,
    labels: list[str],
    title: str,
    output_path: str,
):
    colors = plt.colormaps["tab20"]
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 9
    fig, ax = plt.subplots(figsize=(3, 3))
    for i, label in enumerate(labels):
        color = colors(i)
        precision, recall, _ = precision_recall_curve(df_trues[label], df_probs[label])
        auprc = auc(recall, precision)
        if label == "Enlarged Cardiomediastinum":
            label = "Enl. Card."
        ax.plot(recall, precision, label=f"{label} (AUC = {auprc:.3f})", color=color)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize="xx-small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=1000)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_csv",
        required=True,
        help="Path to input split csv with patient ID, study ID, dicom ID, and split columns",
    )
    parser.add_argument(
        "--metadata_csv",
        required=True,
        help="Path to input metadata csv with patient ID, study ID, dicom ID, and view position columns",
    )
    parser.add_argument(
        "--label_csv",
        required=True,
        help="Path to input label csv with study ID and label columns",
    )
    parser.add_argument(
        "--feature_h5",
        required=True,
        help="Path to input feature h5",
    )
    parser.add_argument(
        "--output_results",
        required=True,
        help="Path to output classifier results (weights, labels, plots)",
    )
    parser.add_argument(
        "--patient_id_col",
        default=DEFAULT_PATIENT_ID_COL,
        help="Name of patient ID column in CSV files",
    )
    parser.add_argument(
        "--study_id_col",
        default=DEFAULT_STUDY_ID_COL,
        help="Name of study ID column in CSV files",
    )
    parser.add_argument(
        "--dicom_id_col",
        default=DEFAULT_DICOM_ID_COL,
        help="Name of dicom ID column in CSV files",
    )
    parser.add_argument(
        "--split_col",
        default=DEFAULT_SPLIT_COL,
        help="Name of split column in CSV files",
    )
    parser.add_argument(
        "--view_col",
        default=DEFAULT_VIEW_COL,
        help="Name of view column in CSV files",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=DEFAULT_LABELS,
        help="Name of labels",
    )
    parser.add_argument(
        "--view_order",
        nargs="+",
        default=DEFAULT_VIEW_ORDER,
        help="Order of views to use",
    )
    parser.add_argument(
        "--feature_key",
        default=DEFAULT_IMG_EMBED_KEY,
        help="Name of image features",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=DEFAULT_MAX_ITER,
        help="Number of iterations for model fitting",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of parallel workers for model fitting",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train_image_classifier(
        split_csv=args.split_csv,
        metadata_csv=args.metadata_csv,
        label_csv=args.label_csv,
        feature_h5=args.feature_h5,
        output_results=args.output_results,
        patient_id_col=args.patient_id_col,
        study_id_col=args.study_id_col,
        dicom_id_col=args.dicom_id_col,
        split_col=args.split_col,
        view_col=args.view_col,
        labels=args.labels,
        view_order=args.view_order,
        feature_key=args.feature_key,
        max_iter=args.max_iter,
        num_workers=args.num_workers,
    )
