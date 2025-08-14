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
from _utils import make_linear_svm_with_probs
from omegaconf import OmegaConf
from pqdm.processes import pqdm
from sklearn.base import ClassifierMixin as Classifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    make_scorer,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier

MODEL_TYPES = Literal["lr", "rf", "svm", "xgb"]
MODEL_MAP = {
    "lr": LogisticRegression,
    "rf": RandomForestClassifier,
    "svm": make_linear_svm_with_probs,
    "xgb": XGBClassifier,
}
HYPERPARAM_T = str | int | float
MODEL_HYPERPARAMS_T = dict[str, HYPERPARAM_T | list[HYPERPARAM_T]]


def make_model(
    *,  # enforce kwargs
    model_type: MODEL_TYPES,
    model_hyperparams: MODEL_HYPERPARAMS_T,
    standard_scale: bool,
) -> Classifier:
    cv_hyperparams = dict()
    static_hyperparams = dict()
    for k, v in model_hyperparams.items():
        if isinstance(v, list) or OmegaConf.is_list(v):
            cv_hyperparams[k] = v
        else:
            static_hyperparams[k] = v

    model_factory = MODEL_MAP[model_type]
    model = model_factory(**static_hyperparams)

    base_has_njobs = "n_jobs" in model.get_params()

    if isinstance(model, CalibratedClassifierCV):
        cv_hyperparams = {f"estimator__{k}": v for k, v in cv_hyperparams.items()}

    if standard_scale:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("classify", model),
            ]
        )
        cv_hyperparams = {f"classify__{k}": v for k, v in cv_hyperparams.items()}

    if len(cv_hyperparams) != 0:
        # if base model doesn't parallelize, do it at the cross val level
        n_jobs = 1 if base_has_njobs else -1
        model = GridSearchCV(
            estimator=model,
            param_grid=cv_hyperparams,
            n_jobs=n_jobs,
            refit=True,
            cv=5,
            scoring=make_scorer(
                roc_auc_score,  # AUROC can take predicted probabilities in the binary case
                greater_is_better=True,
                response_method="predict_proba",
            ),
        )

    assert getattr(model, "predict_proba", None) is not None

    return model


def train_per_label_cls(
    *,  # enforce kwargs
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    model_type: MODEL_TYPES,
    model_hyperparams: MODEL_HYPERPARAMS_T,
    standard_scale: bool,
):
    model = make_model(
        model_type=model_type,
        model_hyperparams=model_hyperparams,
        standard_scale=standard_scale,
    )
    model.fit(X_train, y_train)
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_val = model.predict_proba(X_val)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]

    roc_threshold = get_optimal_threshold(
        trues=y_val,
        probs=y_prob_val,
        method="roc",
    )
    y_pred_roc_train = (y_prob_train > roc_threshold).astype(int)
    y_pred_roc_val = (y_prob_val > roc_threshold).astype(int)
    y_pred_roc_test = (y_prob_test > roc_threshold).astype(int)

    pr_threshold = get_optimal_threshold(
        trues=y_val,
        probs=y_prob_val,
        method="pr",
    )
    y_pred_pr_train = (y_prob_train > pr_threshold).astype(int)
    y_pred_pr_val = (y_prob_val > pr_threshold).astype(int)
    y_pred_pr_test = (y_prob_test > pr_threshold).astype(int)

    return {
        "model": model,
        "y_prob_train": y_prob_train,
        "y_prob_val": y_prob_val,
        "y_prob_test": y_prob_test,
        "y_pred_roc_train": y_pred_roc_train,
        "y_pred_roc_val": y_pred_roc_val,
        "y_pred_roc_test": y_pred_roc_test,
        "y_pred_pr_train": y_pred_pr_train,
        "y_pred_pr_val": y_pred_pr_val,
        "y_pred_pr_test": y_pred_pr_test,
    }


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
    model_type: MODEL_TYPES,
    model_hyperparams: MODEL_HYPERPARAMS_T,
    standard_scale: bool,
    parallelize_labels: int = 1,
):
    assert parallelize_labels > 0
    assert not os.path.exists(output_results)
    os.makedirs(output_results)

    print("Loading Data")
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

    print("Training Models")
    inputs = []
    for label in labels:
        inputs.append(
            {
                "X_train": X_train,
                "X_val": X_val,
                "X_test": X_test,
                "y_train": y_train[label].to_numpy(),
                "y_val": y_val[label].to_numpy(),
                "model_type": model_type,
                "model_hyperparams": model_hyperparams,
                "standard_scale": standard_scale,
            }
        )

    if parallelize_labels != 1:
        results = pqdm(
            inputs,
            train_per_label_cls,
            n_jobs=parallelize_labels,
            argument_type="kwargs",
            desc="Training Label",
            exception_behaviour="immediate",
        )
    else:
        results = [
            train_per_label_cls(**kwargs)
            for kwargs in tqdm(inputs, desc="Training Label")
        ]

    models = dict()
    for label, ret in zip(labels, results):
        models[label] = ret["model"]
        y_prob_train[label] = ret["y_prob_train"]
        y_prob_val[label] = ret["y_prob_val"]
        y_prob_test[label] = ret["y_prob_test"]
        y_pred_roc_train[label] = ret["y_pred_roc_train"]
        y_pred_roc_val[label] = ret["y_pred_roc_val"]
        y_pred_roc_test[label] = ret["y_pred_roc_test"]
        y_pred_pr_train[label] = ret["y_pred_pr_train"]
        y_pred_pr_val[label] = ret["y_pred_pr_val"]
        y_pred_pr_test[label] = ret["y_pred_pr_test"]

    print("Saving Results")

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

    print("Drawing Figures")

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
        "--model_config",
        default="configs/classify/default.yaml",
        help="Classifier model configuration file",
    )
    parser.add_argument(
        "--parallelize_labels",
        type=int,
        default=len(DEFAULT_LABELS),
        help="Number of labels to fit in parallel. Take care to adjust the number of workers used to fit each individual model to avoid process contention",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.model_config)
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
        model_type=config["model_type"],
        model_hyperparams=config["model_hyperparams"],
        standard_scale=config["standard_scale"],
        parallelize_labels=args.parallelize_labels,
    )
