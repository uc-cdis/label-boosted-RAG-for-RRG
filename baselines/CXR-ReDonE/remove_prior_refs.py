import argparse
import json
import os

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
    ret = ret.replace(" ##", "")
    return ret.strip()


def update_json(pipe, in_path, out_path):
    f = open(in_path, "r")
    json_obj = json.load(f)
    f.close()
    for i in tqdm(range(len(json_obj))):
        item = json_obj[i]
        report = item["caption"]
        item["caption"] = remove_priors(pipe, report)
    f = open(out_path, "w")
    json.dump(json_obj, f)
    f.close()


def update_csv(pipe, in_path, out_path):
    df = pd.read_csv(in_path)
    for i in tqdm(range(len(df))):
        if type(df.loc[i, "report"]) == str:
            df.loc[i, "report"] = remove_priors(pipe, df.loc[i, "report"])
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove prior references from report impressions"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="data/",
        help="directory with impression sections from CXR-RePaiR",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="out/",
        help="directory where impression sections are saved",
    )
    args = parser.parse_args()

    pipe = get_pipe()

    # print("Updating JSON")
    # update_json(
    #     pipe,
    #     os.path.join(args.dir, "mimic_train.json"),
    #     os.path.join(args.out, "mimic_train.json"),
    # )

    print("Updating test CSV")
    update_csv(
        pipe,
        os.path.join(args.dir, "mimic_test_impressions.csv"),
        os.path.join(args.out, "mimic_test_impressions.csv"),
    )

    print("Updating train CSV")
    update_csv(
        pipe,
        os.path.join(args.dir, "mimic_train_impressions.csv"),
        os.path.join(args.out, "mimic_train_impressions.csv"),
    )
