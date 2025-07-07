import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULT_IMG_EMBED_KEY = "img_embed"
DEFAULT_IMG_PROJ_KEY = "img_proj"

DEFAULT_PATIENT_ID_COL = "subject_id"
DEFAULT_STUDY_ID_COL = "study_id"
DEFAULT_DICOM_ID_COL = "dicom_id"
DEFAULT_SPLIT_COL = "split"
DEFAULT_VIEW_COL = "ViewPosition"
DEFAULT_LABELS = [
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
DEFAULT_VIEW_ORDER = [
    "PA",
    "AP",
    "LATERAL",
    "LL",
    "AP AXIAL",
    "AP LLD",
    "AP RLD",
    "PA RLD",
    "PA LLD",
    "LAO",
    "RAO",
    "LPO",
    "XTABLE LATERAL",
    "SWIMMERS",
    "",
]
DEFAULT_FINDINGS_COL = "findings"
DEFAULT_IMPRESSION_COL = "impression"


def get_per_study_data(
    *,  # enforce kwargs
    split_csv: str,
    metadata_csv: str,
    label_csv: str | None = None,
    report_csv: str | None = None,
    patient_id_col: str = DEFAULT_PATIENT_ID_COL,
    study_id_col: str = DEFAULT_STUDY_ID_COL,
    dicom_id_col: str = DEFAULT_DICOM_ID_COL,
    split_col: str = DEFAULT_SPLIT_COL,
    view_col: str = DEFAULT_VIEW_COL,
    labels: list[str] = DEFAULT_LABELS,
    view_order: list[str] = DEFAULT_VIEW_ORDER,
    report_cols: list[str] = [DEFAULT_FINDINGS_COL, DEFAULT_IMPRESSION_COL],
    split_remap: dict[str, str] | None = None,
) -> pd.DataFrame:
    split_df = pd.read_csv(split_csv)
    # check all samples per patient are within one split
    assert (split_df.groupby(patient_id_col)[split_col].nunique() == 1).all()

    split_df = split_df[[study_id_col, split_col]].drop_duplicates(study_id_col)
    split_df = split_df.sort_values(study_id_col)
    split_df = split_df.reset_index(drop=True)

    metadata_df = pd.read_csv(metadata_csv)
    metadata_df = metadata_df[[patient_id_col, study_id_col, dicom_id_col, view_col]]

    view_to_index = {v: i for i, v in enumerate(view_order)}
    metadata_df[view_col].fillna("", inplace=True)
    metadata_df = metadata_df.sort_values(view_col, key=lambda v: v.map(view_to_index))
    # stable sort to preserve view order
    metadata_df = metadata_df.sort_values(study_id_col, kind="stable")
    metadata_df = metadata_df.drop_duplicates(study_id_col, keep="first")
    metadata_df = metadata_df.reset_index(drop=True)

    merged_df = split_df.merge(
        metadata_df,
        on=study_id_col,
    )
    cols = [patient_id_col, study_id_col, dicom_id_col, split_col, view_col]

    if label_csv is not None:
        label_df = pd.read_csv(label_csv)
        label_df = label_df[[study_id_col] + labels]
        label_df = label_df.sort_values(study_id_col)
        label_df = label_df.reset_index(drop=True)
        merged_df = merged_df.merge(
            label_df,
            on=study_id_col,
        )
        cols += labels

    if report_csv is not None:
        report_df = pd.read_csv(report_csv)
        report_df = report_df[[study_id_col] + report_cols]
        report_df = report_df.sort_values(study_id_col)
        report_df = report_df.reset_index(drop=True)
        merged_df = merged_df.merge(
            report_df,
            on=study_id_col,
        )
        cols += report_cols

    merged_df = merged_df.sort_values([patient_id_col, study_id_col])
    merged_df = merged_df.sort_values(split_col, kind="stable")
    merged_df = merged_df[cols]
    merged_df = merged_df.reset_index(drop=True)
    if split_remap is not None:
        merged_df[split_col] = merged_df[split_col].replace(split_remap)
    return merged_df


def get_split_features(
    *,  # enforce kwargs
    feature_h5: str,
    feature_key: str,
    sample_df: pd.DataFrame,
    patient_id_col: str = DEFAULT_PATIENT_ID_COL,
    study_id_col: str = DEFAULT_STUDY_ID_COL,
    dicom_id_col: str = DEFAULT_DICOM_ID_COL,
    split_col: str = DEFAULT_SPLIT_COL,
    use_tqdm: bool = True,
) -> dict[str, np.ndarray]:
    ret = dict()
    splits = sample_df[split_col].unique()
    with h5py.File(feature_h5, "r") as h5:
        feature_h5 = h5[feature_key]
        for split in tqdm(splits, desc="Loading Splits", disable=not use_tqdm):
            split_df = sample_df[sample_df[split_col] == split]
            split_features = []
            ids = split_df[[patient_id_col, study_id_col, dicom_id_col]].to_numpy()
            for patient_id, study_id, dicom_id in tqdm(
                ids, desc=f"{split.title()} Samples", disable=not use_tqdm
            ):
                if f"p{patient_id}/s{study_id}/{dicom_id}" in feature_h5:
                    # mimic-cxr style
                    x = feature_h5[f"p{patient_id}/s{study_id}/{dicom_id}"][:]
                else:
                    # chexpertplus style
                    x = feature_h5[f"{patient_id}/{study_id}/{dicom_id}"][:]
                split_features.append(x)
            split_features = np.stack(split_features)
            ret[split] = split_features
    return ret


def get_split_samples(
    *,  # enforce kwargs
    sample_df: pd.DataFrame,
    split_col: str = DEFAULT_SPLIT_COL,
) -> dict[str, pd.DataFrame]:
    ret = dict()
    splits = sample_df[split_col].unique()
    for split in splits:
        split_df = sample_df[sample_df[split_col] == split].reset_index(drop=True)
        split_df = split_df.copy()
        ret[split] = split_df
    return ret
