import argparse

import pandas as pd
import torch
from bert_score import BERTScorer
from tqdm import tqdm


def pred(model, premise, hypothesis):
    _, _, f1 = model.score([premise], [hypothesis])
    return f1


def main(args):
    model = BERTScorer(
        model_type="distilroberta-base",
        batch_size=256,
        lang="en",
        rescale_with_baseline=True,
    )
    ve_output = pd.read_csv(args.input_path)
    filtered_doc = []
    threshold = 0.5

    for doc in tqdm(range(len(ve_output))):
        reports = ve_output["Report Impression"][doc].split(args.delimiter)
        report = reports[0]
        added = 1
        for i in range(1, len(reports)):
            bertscore_f1 = pred(model, report, reports[i])
            if bertscore_f1 < threshold:
                report += " " + reports[i]
                added += 1
            if added >= args.topk:
                break
        filtered_doc.append(report)
    ve_output["filtered"] = filtered_doc
    ve_output.to_csv(args.save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        default="../ALBEF/example.csv",
        help="path to the output of the VE module",
    )
    parser.add_argument("--save_path", default="example_bertscore_3.csv")
    parser.add_argument("--delimiter", default="[SEP]")
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()
    main(args)
