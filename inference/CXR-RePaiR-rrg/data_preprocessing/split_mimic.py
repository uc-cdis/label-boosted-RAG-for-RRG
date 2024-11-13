import argparse
import glob
import gzip
import os

import pandas as pd
from tqdm import tqdm

"""Split MIMIC-CXR reports to train and test splits according to MIMIC-CXR-JPG."""


def get_mimic_train_test_split(mimic_cxr_jpg_split_path):
    """Splits MIMIC-CXR DICOM IDs, study IDs, subject IDs into train/test sets.
    Splits according to MIMIC-CXR-JPG recommended splits. We combine DICOM IDs,
    study IDs and subject IDs associated with `train` and `validate` into the
    train set, and those with `test` into the test set.
    Args:
      mimic_cxr_jpg_split_path: Full path of mimic split (ending in mimic-cxr-2.0.0-split.csv.gz)
    Returns:
      Train and test lists of DICOM IDs, study IDs and subject IDs (with
      duplicates).
    """
    train_ids, test_ids = [], []

    with gzip.open(mimic_cxr_jpg_split_path, "rb") as f:
        skip_header = True

        for sample_raw in f:
            if skip_header:
                skip_header = False
                continue
            sample = sample_raw.decode("ascii")[:-1]  # Remove newline.
            dicom_id, study_id, subject_id, split = sample.split(",")
            if split == "train" or split == "validate":
                train_ids.append((dicom_id, study_id, subject_id))
            elif split == "test":
                test_ids.append((dicom_id, study_id, subject_id))
            else:
                raise Exception(f"Unknown split label {split}.")

        print(f"train study IDs: {len(train_ids)}\n" f"test study IDs: {len(test_ids)}")
        assert len(set(train_ids).intersection(set(test_ids))) == 0

    return train_ids, test_ids


def generate_mimic_train_test_csv(report_files_dir, split_ids, csv_path):
    """Generates MIMIC-CXR reports of `split_ids` as a CSV file.
    The CSV contains [DICOM ID, study ID, subject ID, full & raw report text].
    Args:
      split_ids: List of (DICOM ID, study ID, subject ID) tuples.
      csv_path: Path to save the CSV file.
    """
    reports = []
    report_files = glob.glob(os.path.join(report_files_dir, "*/*/*.txt"))

    for dicom_id, study_id, subject_id in tqdm(split_ids):
        report_subpath = f"p{subject_id[:2]}/p{subject_id}/s{study_id}.txt"
        report_path = os.path.join(report_files_dir, report_subpath)
        if report_path in report_files:
            with open(report_path, "r") as f:
                full_report = f.read()
                reports.append((dicom_id, study_id, subject_id, full_report))
        else:
            print(f"Report {report_path} not found.")

    df = pd.DataFrame(reports, columns=["dicom_id", "study_id", "subject_id", "report"])
    df.to_csv(csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create csvs for the full reports of mimic train and test"
    )

    parser.add_argument(
        "--report_files_dir",
        type=str,
        required=True,
        help="mimic files directory containing all reports",
    )
    parser.add_argument(
        "--split_path",
        type=str,
        required=True,
        help="path to split file in mimic-cxr-jpg",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="directory to store train and test splits",
    )

    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    train_path = os.path.join(args.out_dir, "mimic_train_full.csv")
    test_path = os.path.join(args.out_dir, "mimic_test_full.csv")

    train_ids, test_ids = get_mimic_train_test_split(args.split_path)
    generate_mimic_train_test_csv(args.report_files_dir, train_ids, train_path)
    generate_mimic_train_test_csv(args.report_files_dir, test_ids, test_path)
