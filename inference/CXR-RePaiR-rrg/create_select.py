import argparse
import os

import pandas as pd

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


def main():
    parser = argparse.ArgumentParser(
        description="Label a csv file containing radiology reports"
    )
    parser.add_argument(
        "-dir",
        "--dir",
        required=True,
        type=str,
        help="directory containing CXR-RePaiR-1 and CXR-RePaiR-6",
    )
    # parser.add_argument('-n_1', '--name_1', type=str, nargs='?', help='name for 1 sentence folder')
    # parser.add_argument('-n_6', '--name_6', type=str, nargs='?', help='name for 6 sentence folder')
    # parser.add_argument('-out', '--out', type=str, nargs='?', help='output name')
    args = parser.parse_args()

    pred_labels_1 = pd.read_csv(
        args.dir + "CXR-RePaiR-1/labeled_reports.csv", index_col=False
    ).fillna(0)[useful_labels]
    pred_labels_1 = pred_labels_1.to_numpy()
    pred_reports_1 = pd.read_csv(
        args.dir + "CXR-RePaiR-1/generated_reports.csv", index_col=False
    )
    pred_reports_6 = pd.read_csv(
        args.dir + "CXR-RePaiR-6/generated_reports.csv", index_col=False
    )
    # pred_labels_1 = pd.read_csv("out/{}/labeled_reports.csv".format(args.name_1), index_col=False).fillna(0)[useful_labels]
    # pred_labels_1 = pred_labels_1.to_numpy()
    # pred_reports_1 = pd.read_csv("out/{}/{}_generated_reports.csv".format(args.name_1, args.name_1), index_col=False)
    # pred_reports_6 = pd.read_csv("out/{}/{}_generated_reports.csv".format(args.name_6, args.name_6), index_col=False)

    combined_df = pred_reports_1.copy()
    for i, row in combined_df.iterrows():
        if pred_labels_1[i][useful_labels.index("No Finding")] != 1.0:
            combined_df.at[i, "report"] = pred_reports_6.at[i, "report"]

    out_path = args.dir + "CXR-RePaiR-Select/generated_reports.csv"
    if not os.path.exists(args.dir + "CXR-RePaiR-Select"):
        os.makedirs(args.dir + "CXR-RePaiR-Select")
    combined_df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
