import argparse
import os

import nltk

nltk.download("punkt_tab")

import nltk.data
import pandas as pd


def section_start(lines, section=" IMPRESSION"):
    """Finds line index that is the start of the section."""
    for idx, line in enumerate(lines):
        if line.startswith(section):
            return idx
    return -1


def generate_whole_report_impression_csv(df, split, dir):
    """Generates a csv containing report impressions."""
    df_imp = df.copy()
    for index, row in df_imp.iterrows():
        report = row["report"].splitlines()

        impression_idx = section_start(report)
        impression_and_findings_idx = section_start(
            report, section=" FINDINGS AND IMPRESSION:"
        )
        seperator = ""
        if impression_idx != -1:
            impression = (
                seperator.join(report[impression_idx:])
                .replace("IMPRESSION:", "")
                .replace("\n", "")
                .strip()
            )
        elif impression_and_findings_idx != -1:
            impression = (
                seperator.join(report[impression_and_findings_idx:])
                .replace("FINDINGS AND IMPRESSION:", "")
                .replace("\n", "")
                .strip()
            )
        else:
            impression = ""

        df_imp.at[index, "report"] = impression

    out_name = f"mimic_{split}_impressions.csv"
    out_path = os.path.join(dir, out_name)
    df_imp.to_csv(out_path, index=False)


def generate_sentence_level_impression_csv(df, split, dir, tokenizer):
    """Generates a csv containing all impression sentences."""
    df_imp = []
    for index, row in df.iterrows():
        report = row["report"].splitlines()

        impression_idx = section_start(report)
        impression_and_findings_idx = section_start(
            report, section=" FINDINGS AND IMPRESSION:"
        )
        seperator = ""
        if impression_idx != -1:
            impression = (
                seperator.join(report[impression_idx:])
                .replace("IMPRESSION:", "")
                .replace("\n", "")
                .strip()
            )
        elif impression_and_findings_idx != -1:
            impression = (
                seperator.join(report[impression_and_findings_idx:])
                .replace("FINDINGS AND IMPRESSION:", "")
                .replace("\n", "")
                .strip()
            )
        else:
            impression = ""

        for sent_index, sent in enumerate(split_sentences(impression, tokenizer)):
            df_imp.append(
                [row["dicom_id"], row["study_id"], row["subject_id"], sent_index, sent]
            )

    df_imp = pd.DataFrame(
        df_imp, columns=["dicom_id", "study_id", "subject_id", "sentence_id", "report"]
    )

    out_name = f"mimic_{split}_sentence_impressions.csv"
    out_path = os.path.join(dir, out_name)
    df_imp.to_csv(out_path, index=False)


def split_sentences(report, tokenizer):
    """Splits sentences by periods and removes numbering and nans."""
    sentences = []
    if not (isinstance(report, float) and math.isnan(report)):
        for sentence in tokenizer.tokenize(report):
            try:
                float(sentence)  # Remove numbering
            except ValueError:
                sentences.append(sentence)
    return sentences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract the impression section and generate csvs for report level and sentence level."
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="directory where train and test report reports are stored and where impression sections will be stored",
    )
    args = parser.parse_args()

    train_path = os.path.join(args.dir, "mimic_train_full.csv")
    test_path = os.path.join(args.dir, "mimic_test_full.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # whole reports
    generate_whole_report_impression_csv(train_df, "train", args.dir)
    generate_whole_report_impression_csv(test_df, "test", args.dir)

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    # sentences
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    generate_sentence_level_impression_csv(train_df, "train", args.dir, tokenizer)
