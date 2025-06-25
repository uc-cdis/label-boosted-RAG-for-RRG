import argparse
import os
from typing import Literal, get_args

import evaluate
import pandas as pd
from _data import DEFAULT_STUDY_ID_COL
from f1chexbert import F1CheXbert
from radgraph import F1RadGraph
from sklearn.metrics import f1_score
from tqdm import tqdm

METRIC = Literal["bleu4", "rougeL", "bertscore", "f1radgraph", "f1chexbert"]
DEFAULT_METRICS = list(get_args(METRIC))
DEFAULT_REF_COL = "actual_text"
DEFAULT_HYP_COL = "generated_text"


def fill_empty(xs: pd.Series) -> list[str]:
    # TODO parameterize NaN string fill
    temp = xs.copy()
    mask = temp.isna() | (temp.str.strip() == "")
    temp[mask] = "EMTPY"
    return temp.to_list()


def evaluate_generations(
    report_csv: str,
    output_csv: str,
    metrics: list[METRIC] = DEFAULT_METRICS,
    ref_col: str = DEFAULT_REF_COL,
    hyp_col: str = DEFAULT_HYP_COL,
    study_id_col: str = DEFAULT_STUDY_ID_COL,
):
    if os.path.exists(output_csv):
        print("File Exists, Exiting")
        return
    report_df = pd.read_csv(report_csv)
    refs = fill_empty(report_df[ref_col])
    hyps = fill_empty(report_df[hyp_col])

    all_results = [report_df[study_id_col]]
    for metric in tqdm(metrics):
        print(f"Computing metric: {metric}")
        if metric == "bleu4":
            bleu = evaluate.load("bleu")
            bleu_results = []
            for ref, hyp in zip(refs, hyps):
                temp = bleu.compute(predictions=[hyp], references=[ref])
                bleu_results.append(temp)
            results = pd.DataFrame(bleu_results)["bleu"]
        elif metric == "rougeL":
            rouge = evaluate.load("rouge")
            rouge_results = rouge.compute(
                predictions=hyps,
                references=refs,
                rouge_types=["rougeL"],
                use_aggregator=False,
            )["rougeL"]
            results = pd.Series(rouge_results)
        elif metric == "bertscore":
            bertscore = evaluate.load("bertscore")
            bert_results = bertscore.compute(
                predictions=hyps,
                references=refs,
                lang="en",
            )
            results = pd.Series(bert_results["f1"])
        elif metric == "f1radgraph":
            f1radgraph = F1RadGraph(
                model_type="radgraph-xl",
                reward_level="partial",
            )
            _, f1radgraph_results, hyp_annots, ref_annots = f1radgraph(
                hyps=hyps, refs=refs
            )
            results = pd.Series(f1radgraph_results)
            hyp_annots = pd.Series(hyp_annots, name="generated_radgraph")
            ref_annots = pd.Series(ref_annots, name="actual_radgraph")
            all_results.append(hyp_annots)
            all_results.append(ref_annots)
        elif metric == "f1chexbert":
            f1chexbert = F1CheXbert()
            refs_chexbert = [f1chexbert.get_label(l.strip()) for l in refs]
            hyps_chexbert = [f1chexbert.get_label(l.strip()) for l in hyps]
            f1chexbert_results = [
                f1_score(r, h) for r, h in zip(refs_chexbert, hyps_chexbert)
            ]
            results = pd.Series(f1chexbert_results)
            hyps_chexbert = pd.Series(hyps_chexbert, name="generated_chexbert")
            refs_chexbert = pd.Series(refs_chexbert, name="actual_chexbert")
            all_results.append(hyps_chexbert)
            all_results.append(refs_chexbert)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        results.name = metric
        all_results.append(results)
    all_results = pd.concat(all_results, axis="columns")
    all_results.to_csv(output_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report_csv",
        required=True,
        help="Path to input report csv with study ID and report columns",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Path to save output evaluation metrics",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        choices=DEFAULT_METRICS,
        help="Set of metrics to evlauate generations",
    )
    parser.add_argument(
        "--ref_col",
        default=DEFAULT_REF_COL,
        help="Name of reference text column in CSV files",
    )
    parser.add_argument(
        "--hyp_col",
        default=DEFAULT_HYP_COL,
        help="Name of generated text column in CSV files",
    )
    parser.add_argument(
        "--study_id_col",
        default=DEFAULT_STUDY_ID_COL,
        help="Name of study ID column in CSV files",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    evaluate_generations(
        report_csv=args.report_csv,
        output_csv=args.output_csv,
        metrics=args.metrics,
        ref_col=args.ref_col,
        hyp_col=args.hyp_col,
        study_id_col=args.study_id_col,
    )
