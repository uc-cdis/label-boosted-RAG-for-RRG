import pandas as pd

reports_set = "/deep/group/data/med-data/mimic-cxr-jpg-split/mimic_train_impressions_final.csv"  # set that was used for training
sentences_set = (
    "/deep/group/data/med-data/mimic-cxr-jpg-split/mimic_train_sentence_impressions.csv"
)


def main():
    original_set = pd.read_csv(sentences_set, index_col=False)
    original_set["report_lower"] = original_set["report"].str.lower()
    unique_reports = original_set.drop_duplicates(subset="report_lower")
    unique_reports = unique_reports.drop(columns="report_lower")
    print(unique_reports)
    unique_reports.to_csv(
        "/deep/group/data/med-data/mimic-cxr-jpg-split/mimic_train_sentence_impressions_unique_corpus.csv",
        index=False,
    )
    # unique_reports.to_csv('/deep/group/data/med-data/mimic-cxr-jpg-split/mimic_train_impressions_unique_corpus.csv', index=False)


if __name__ == "__main__":
    main()
