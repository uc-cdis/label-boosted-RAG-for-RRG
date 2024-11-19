import argparse
import os

# fmt: off
# isort: off
from _data import (
    DEFAULT_DICOM_ID_COL as dicom_id_col,
    DEFAULT_FINDINGS_COL as findings_col,
    DEFAULT_IMPRESSION_COL as impression_col,
    DEFAULT_LABELS as labels,
    DEFAULT_PATIENT_ID_COL as patient_id_col,
    DEFAULT_SPLIT_COL as split_col,
    DEFAULT_STUDY_ID_COL as study_id_col,
    DEFAULT_VIEW_COL as view_col,
    DEFAULT_VIEW_ORDER as view_order,
    get_per_study_data,
    get_split_samples,
)
# isort: on
# fmt: on


def extract_image_path(data_dir, row):
    subject_id = str(row["subject_id"])
    study_id = str(row["study_id"])
    dicom_id = str(row["dicom_id"])
    return os.path.join(
        data_dir,
        f"p{subject_id[:2]}",
        f"p{subject_id}",
        f"s{study_id}",
        f"{dicom_id}.jpg",
    )


def main(
    *,  # enforce kwargs
    data_dir: str,
    split_csv: str,
    metadata_csv: str,
    report_csv: str,
    true_label_csv: str,
    output_dir: str,
):
    split_remap = {
        "train": "retrieval",
        "validate": "retrieval",
        "test": "inference",
    }

    inference_df = get_per_study_data(
        split_csv=split_csv,
        metadata_csv=metadata_csv,
        label_csv=true_label_csv,
        report_csv=report_csv,
        patient_id_col=patient_id_col,
        study_id_col=study_id_col,
        dicom_id_col=dicom_id_col,
        split_col=split_col,
        view_col=view_col,
        labels=labels,
        view_order=view_order,
        report_cols=[findings_col, impression_col],
        split_remap=split_remap,
    )

    findings_df = (
        inference_df[inference_df[findings_col].notna()].reset_index(drop=True).copy()
    )
    impression_df = (
        inference_df[inference_df[impression_col].notna()].reset_index(drop=True).copy()
    )

    findings_samples = get_split_samples(
        sample_df=findings_df,
        split_col=split_col,
    )["inference"]
    impression_samples = get_split_samples(
        sample_df=impression_df,
        split_col=split_col,
    )["inference"]

    findings_samples["dicom_path"] = findings_samples.apply(
        lambda r: extract_image_path(data_dir, r),
        axis=1,
    )
    impression_samples["dicom_path"] = impression_samples.apply(
        lambda r: extract_image_path(data_dir, r),
        axis=1,
    )

    findings_samples.to_csv(
        os.path.join(output_dir, "findings.csv"),
        index=False,
    )
    impression_samples.to_csv(
        os.path.join(output_dir, "impression.csv"),
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/opt/gpudata/mimic-cxr/files/",
    )
    parser.add_argument(
        "--split_csv",
        default="/opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv",
    )
    parser.add_argument(
        "--metadata_csv",
        default="/opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv",
    )
    parser.add_argument(
        "--report_csv",
        default="/opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv",
    )
    # Label values unused but potentially alters data subset
    # Equivalent to use true or predicted label for subsetting
    parser.add_argument(
        "--true_label_csv",
        default="/opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="/opt/gpudata/rrg-data-2/baselines",
    )
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        split_csv=args.split_csv,
        metadata_csv=args.metadata_csv,
        report_csv=args.report_csv,
        true_label_csv=args.true_label_csv,
        output_dir=args.output_dir,
    )
