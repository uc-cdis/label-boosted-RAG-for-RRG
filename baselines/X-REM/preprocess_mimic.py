#!/usr/bin/env python
# coding: utf-8

import argparse
import gzip
import json
import os

import pandas as pd
from tqdm import tqdm


def main(args):
    impressions_train_df = pd.read_csv(args.impressions_train_path)
    impressions_test_df = pd.read_csv(args.impressions_test_path)
    impressions_dfs = {
        "train": impressions_train_df,
        "validate": impressions_train_df,
        "test": impressions_test_df,
    }
    mimic_cxr_jpg_split_path = os.path.join(
        args.data_dir, "raw_jpg", "mimic-cxr-2.0.0-split.csv.gz"
    )

    train_json = []
    val_json = []
    test_json = []

    # directory structure is p10 / p<subject id> / s<study id> / images
    with gzip.open(mimic_cxr_jpg_split_path, "rb") as f:
        skip_header = True

        for sample_raw in tqdm(f):
            if skip_header:
                skip_header = False
                continue
            sample = sample_raw.decode("ascii")[:-1]  # Remove newline.
            dicom_id, study_id, subject_id, split = sample.split(
                ","
            )  # extract one item

            # construct the path to the image
            patient_folder2 = "".join(["p", subject_id])
            patient_folder1 = patient_folder2[:3]
            study_folder = "".join(["s", study_id])
            file_name = "".join([dicom_id, ".jpg"])
            img_path = os.path.join(
                args.data_dir,
                "files",
                patient_folder1,
                patient_folder2,
                study_folder,
                file_name,
            )

            # access the impression part of the report
            impressions_df = impressions_dfs[split]
            single_df = impressions_df[impressions_df["dicom_id"] == dicom_id]["report"]
            if len(single_df) == 0:  # error, dicom_id not found, then skip
                continue

            impression_str = single_df.iat[0]
            if pd.isna(impression_str):  # no impression found
                continue

            json_item = {
                "caption": impression_str,
                "image": img_path,
                "dicom_id": dicom_id,
                "study_id": study_id,
            }

            if split == "train":
                train_json.append(json_item)
            elif split == "validate":
                val_json.append(json_item)
            elif split == "test":
                test_json.append(json_item)
            else:
                raise Exception(f"Unknown split label {split}.")

    jsons = {"train": train_json, "val": val_json, "test": test_json}

    splits_options = ["train", "val", "test"]
    for split_op in splits_options:
        json_str = json.dumps(jsons[split_op])
        fname = "".join([args.out_dir, "mimic_", split_op, ".json"])
        with open(fname, "w") as outfile:
            outfile.write(json_str)
        print(split_op, len(jsons[split_op]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="path to MIMIC-CXR")
    parser.add_argument(
        "--impressions_train_path",
        help="path to mimic_train_impressions.csv created by CXR-RePaiR",
    )
    parser.add_argument(
        "--impressions_test_path",
        help="path to mimic_test_impressions.csv created by CXR-RePaiR",
    )
    parser.add_argument("--out_dir", help="directory to store the outputs")
    args = parser.parse_args()
    main(args)
