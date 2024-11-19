import argparse

import pandas as pd
import torch
from M2TransNLI import M2TransNLI
from tqdm import tqdm


def softmax(x):
    s = torch.exp(output)
    return s / torch.sum(s)


def pred(nli_model, premise, hypothesis):
    ret = nli_model.predict([premise], [hypothesis])[1][0]
    assert ret == "entailment" or ret == "contradiction" or ret == "neutral"
    return ret


def main(args):
    model = M2TransNLI.load_model(args.m2trans_nli_model_path)
    nli_model = M2TransNLI(model)
    ve_output = pd.read_csv(args.input_path)
    filtered_doc = []

    for doc in tqdm(range(len(ve_output))):
        reports = ve_output["Report Impression"][doc].split(args.delimiter)
        report = reports[0]
        added = 1
        for i in range(1, len(reports)):
            nli_label = pred(nli_model, report, reports[i])
            if nli_label == "neutral":
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
        "--m2trans_nli_model_path",
        required=True,
        help="path to model_medrad_19k directory",
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="path to the output of the VE module",
    )
    parser.add_argument("--save_path", default="example_m2trans_nli.csv")
    parser.add_argument("--delimiter", default="[SEP]")
    parser.add_argument("--topk", type=int, default=2)
    args = parser.parse_args()
    main(args)
