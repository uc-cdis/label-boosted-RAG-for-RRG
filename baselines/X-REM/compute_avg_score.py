import argparse
import glob

import numpy as np
import pandas as pd


def main(fpath):
    sent = pd.read_csv(fpath)
    keys = ["bleu_score", "bertscore", "semb_score", "radgraph_combined"]
    print("\nComputing scores for: " + fpath + "\n")
    for k in keys:
        print(k, np.average(sent[k]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpath", default="scores.csv")
    args = parser.parse_args()
    main(args.fpath)
