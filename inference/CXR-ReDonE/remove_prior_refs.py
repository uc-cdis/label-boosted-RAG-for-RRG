import argparse
import json

import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


def get_pipe():
    model_name = "rajpurkarlab/gilbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    pipe = pipeline(
        task="token-classification",
        model=model.to("cpu"),
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )
    return pipe


def remove_priors(pipe, report):
    ret = ""
    for sentence in report.split("."):
        if sentence and not sentence.isspace():
            p = pipe(sentence)
            string = ""
            for item in p:
                if item["entity_group"] == "KEEP":
                    string += item["word"] + " "
            ret += (
                string.strip().replace("redemonstrate", "demonstrate").capitalize()
                + ". "
            )
    return ret.strip()


def update_json(pipe, path):
    f = open(path, "r")
    json_obj = json.load(f)
    f.close()
    for i in tqdm(range(len(json_obj))):
        item = json_obj[i]
        report = item["caption"]
        item["caption"] = remove_priors(pipe, report)
    f = open(path, "w")
    json.dump(json_obj, f)
    f.close()


def update_csv(pipe, path):
    df = pd.read_csv(path)
    for i in tqdm(range(len(df))):
        if type(df.loc[i, "report"]) == str:
            df.loc[i, "report"] = remove_priors(pipe, df.loc[i, "report"])
    df.to_csv(path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove prior references from report impressions"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="data/",
        help="directory where impression sections are stored",
    )
    args = parser.parse_args()

    pipe = get_pipe()

    train_json = args.dir + "mimic_train.json"
    train_csv = args.dir + "mimic_train_impressions.csv"
    test_csv = args.dir + "mimic_test_impressions.csv"

    print("Updating JSON")
    update_json(pipe, train_json)

    print("Updating test CSV")
    update_csv(pipe, test_csv)

    print("Updating train CSV")
    update_csv(pipe, train_csv)
