import argparse

import pandas as pd


# CXR-Metric computes the scores based on 2,192 samples from the mimic-cxr test set that consists of 3,678
# This function selects the respective 2,192 reports
def main(fpath, opath, use_ve):
    if use_ve:
        filt = "filtered"
    else:
        filt = "Report Impression"
    df = pd.read_csv(fpath)
    test = pd.read_csv("../data/mimic_test_impressions.csv")
    gt = pd.read_csv("../data/cxr2_generated_appa.csv")[
        ["dicom_id", "study_id"]
    ]  # contains the dicom ids of the 2,192 imgs
    gt["report"] = [None] * len(gt)
    pred = df[filt]
    pred = pd.concat([test[["dicom_id"]], pred], axis=1)
    pred = pred.set_index("dicom_id")
    for idx, row in gt.iterrows():
        dicom_id, study_id, _ = row
        gt["report"][idx] = pred[filt][dicom_id]
    gt.to_csv(opath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpath", default="../ifcc/example_m2trans_nli.csv")
    parser.add_argument("--opath", default="pred.csv")
    parser.add_argument("--use_ve", type=bool, default=False)
    args = parser.parse_args()

    if not args.use_ve:
        args.fpath = "../ALBEF/example.csv"

    main(args.fpath, args.opath, args.use_ve)
