import os
from collections import defaultdict
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator


def check_duplicate_runs(result_dir):
    print("===== Checking duplicate runs are equivalent =====\n")
    for dataset in ["mimic-cxr", "chexpertplus"]:
        for section in ["findings", "impression"]:
            print(f"-------- {dataset}/{section} --------")
            experiments = get_experiments_metadata(
                dataset=dataset,
                section=section,
                result_dir=result_dir,
            )
            count = defaultdict(list)
            for exp_name, (exp_dir, trials) in experiments.items():
                for trial_name, trial_file in trials:
                    count[trial_file].append(os.path.join(exp_dir, trial_file))

            cols = ["bleu4", "rougeL", "bertscore", "f1radgraph", "f1chexbert"]
            dupes = {k: v for k, v in count.items() if len(v) > 1}
            print("Duplicates:\n")
            for k, vs in dupes.items():
                print(k)
                group_dfs = []
                for v in vs:
                    df = pd.read_csv(v)
                    group_dfs.append(df)
                ref = group_dfs[0]
                print(f"--- ref: {vs[0]}")
                for df, v in zip(group_dfs[1:], vs[1:]):
                    print(f"--- cmp: {v}")
                    assert (ref["study_id"] == df["study_id"]).all()
                    assert np.isclose(ref[cols], df[cols]).all()
                print()
    print("===== All duplicate runs are equivalent! =====")


def get_experiment_results(
    *,  # enforce kwargs
    exp_dir: str,
    exp_trials: list[tuple[str, str]],
    normalize_bertscore_lang: str | None = None,
) -> list[pd.DataFrame]:
    if normalize_bertscore_lang is not None:
        # assumes that bertscores are F1
        from bert_score import BERTScorer

        scorer = BERTScorer(lang=normalize_bertscore_lang, device="cpu")
        baseline_f1 = scorer.baseline_vals[-1].numpy()

    trial_dfs = []
    for _, trial_file in exp_trials:
        trial_df = pd.read_csv(os.path.join(exp_dir, trial_file))

        if normalize_bertscore_lang is not None:
            # https://github.com/Tiiiger/bert_score/blob/master/journal/rescale_baseline.md
            x = trial_df["bertscore"]
            trial_df["bertscore-original"] = x.copy()
            trial_df["bertscore"] = (x - baseline_f1) / (1 - baseline_f1)

        trial_dfs.append(trial_df)

    # intersection of study ids
    ids = set.intersection(*map(set, [df["study_id"] for df in trial_dfs]))
    ids = sorted(list(ids))
    trial_dfs = [df.set_index("study_id").loc[ids].reset_index() for df in trial_dfs]
    return trial_dfs


def plot_experiment(
    *,  # enforce kwargs
    section: str,  # for title only
    exp_name: str,
    exp_trials: list[tuple[str, str]],
    trial_dfs: list[pd.DataFrame],
    metrics: list[str],
) -> plt.Figure:
    # setup dataframe for seaborn barplot
    melted_results = []
    for trial_df, (trial_name, _) in zip(trial_dfs, exp_trials):
        trial_df = trial_df.melt(id_vars="study_id", var_name="metric")
        trial_df[exp_name] = trial_name
        melted_results.append(trial_df)
    df = pd.concat(melted_results, ignore_index=True)

    # filter metrics for plotting
    df = df[df["metric"].isin(metrics)]

    # setup seaborn barplot parameters
    x = "metric"
    y = "value"
    hue = exp_name
    hue_order = [trial_name for trial_name, _ in exp_trials]
    palette = [MODEL2COLOR[trial_file] for _, trial_file in exp_trials]
    order = metrics
    if exp_name == "Literature":
        # only do stats tests compared to ours if evaluating literature models
        pairs = [
            ((metric, "LaB-RAG"), (metric, n2))
            for metric in metrics
            for n2 in hue_order[1:]
        ]
    else:
        # otherwise do all pairwise comparisons of stats tests
        pairs = [
            ((metric, n1), (metric, n2))
            for metric in metrics
            for i, n1 in enumerate(hue_order)
            for n2 in hue_order[i + 1 :]
        ]

    # do plotting
    extra_room = "bertscore" in metrics
    if extra_room:
        fig, ax = plt.subplots(figsize=(6, 3))
    else:
        fig, ax = plt.subplots(figsize=(3, 3))
    barplot = sns.barplot(
        df,
        x=x,
        y=y,
        order=order,
        hue=hue,
        hue_order=hue_order,
        palette=palette,
        ax=ax,
        saturation=1,
        zorder=15,
        errorbar="se",
        err_kws={
            "zorder": 25,
            "linewidth": 1,
            "alpha": 1,
        },
        width=0.15 * len(hue_order),
    )
    annot = Annotator(
        ax,
        pairs,
        data=df,
        x=x,
        y=y,
        order=order,
        hue=hue,
        hue_order=hue_order,
        palette=palette,
        width=0.15 * len(hue_order),
    )
    annot._pvalue_format.fontsize = 9
    annot.configure(
        test="t-test_paired",
        comparisons_correction="Bonferroni",
        hide_non_significant=True,
        line_height=0.04,
        text_offset=-3,
        line_offset=10000,
        line_offset_to_group=0.1,
        line_width=0.75,
        pvalue_thresholds=[[0.05, "*"], [1, "ns"]],
    )
    _, annotations = annot.apply_test().annotate(line_offset=10000)

    # format plot
    ax.set_xlabel("")
    ax.set_ylabel("")
    if extra_room:
        ax.set_ylim([-0.05, 1.55])
    else:
        ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([-0.5, len(metrics) - 0.5])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(which="major", axis="y", zorder=0)
    ax.set_title(f"{section.title()}, N={len(trial_dfs[0])}", fontsize=10)
    legend = ax.legend(title=None, loc="upper left")
    legend.set_zorder(10)

    fig.tight_layout()
    return fig


def get_experiments_metadata(
    *,  # enforce kwargs
    dataset: Literal["mimic-cxr", "chexpertplus"],
    section: Literal["findings", "impression"],
    result_dir: str,
):
    if dataset == "mimic-cxr":
        emb_type = "BioViL-T"
        label_type = "mimic-cxr-biovilt-pred"
        dataset_dir = "exp-mimic"
    elif dataset == "chexpertplus":
        emb_type = "GLoRIA"
        label_type = "chexpertplus-gloria-pred"
        dataset_dir = "exp-chexpertplus"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Formatted as:
    # Experiment name:
    #     Experiment directory
    #     Trials:
    #         Trial name
    #         Trial file
    experiments = {
        "Core": (
            os.path.join(result_dir, dataset_dir, f"exp-{section}", "exp-core"),
            [
                (
                    "Standard RAG",
                    f"{section}_top-5_{label_type}-label_no-filter_naive_Mistral-7B-Instruct-v0.3_METRICS.csv",
                ),
                (
                    "Label Filter only",
                    f"{section}_top-5_{label_type}-label_exact_naive_Mistral-7B-Instruct-v0.3_METRICS.csv",
                ),
                (
                    "Label Format only",
                    f"{section}_top-5_{label_type}-label_no-filter_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
                ),
                (
                    "LaB-RAG",
                    f"{section}_top-5_{label_type}-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
                ),
            ],
        ),
        "Filter": (
            os.path.join(result_dir, dataset_dir, f"exp-{section}", "exp-filter"),
            [
                (
                    "No-filter",
                    f"{section}_top-5_{label_type}-label_no-filter_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
                ),
                (
                    "Exact",
                    f"{section}_top-5_{label_type}-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
                ),
                (
                    "Partial",
                    f"{section}_top-5_{label_type}-label_partial_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
                ),
            ],
        ),
        "Prompt": (
            os.path.join(result_dir, dataset_dir, f"exp-{section}", "exp-prompt"),
            [
                (
                    "Naive",
                    f"{section}_top-5_{label_type}-label_exact_naive_Mistral-7B-Instruct-v0.3_METRICS.csv",
                ),
                (
                    "Simple",
                    f"{section}_top-5_{label_type}-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
                ),
                (
                    "Verbose",
                    f"{section}_top-5_{label_type}-label_exact_verbose_Mistral-7B-Instruct-v0.3_METRICS.csv",
                ),
                (
                    "Instruct",
                    f"{section}_top-5_{label_type}-label_exact_instruct_Mistral-7B-Instruct-v0.3_METRICS.csv",
                ),
            ],
        ),
        "Language Model": (
            os.path.join(result_dir, dataset_dir, f"exp-{section}", "exp-llm"),
            [
                (
                    "Mistral-v1",
                    f"{section}_top-5_{label_type}-label_exact_simple_Mistral-7B-Instruct-v0.1_METRICS.csv",
                ),
                (
                    "BioMistral",
                    f"{section}_top-5_{label_type}-label_exact_simple_BioMistral-7B_METRICS.csv",
                ),
                (
                    "Mistral-v3",
                    f"{section}_top-5_{label_type}-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
                ),
            ],
        ),
        "Embedding Model": (
            os.path.join(result_dir, dataset_dir, f"exp-{section}", "exp-embedding"),
            [
                (
                    emb_type,
                    f"{section}_top-5_{label_type}-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
                ),
                (
                    "ResNet50",
                    f"{section}_top-5_{dataset}-resnet50-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
                ),
            ],
        ),
        "Label Quality": (
            os.path.join(result_dir, dataset_dir, f"exp-{section}", "exp-true-label"),
            [
                (
                    "True",
                    f"{section}_top-5_true-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
                ),
                (
                    "Predicted",
                    f"{section}_top-5_{label_type}-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
                ),
            ],
        ),
        "Retrieved Samples": (
            os.path.join(result_dir, dataset_dir, f"exp-{section}", "exp-top-k"),
            [
                (
                    "3",
                    f"{section}_top-3_{label_type}-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
                ),
                (
                    "5",
                    f"{section}_top-5_{label_type}-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
                ),
                (
                    "10",
                    f"{section}_top-10_{label_type}-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
                ),
            ],
        ),
    }
    if dataset == "mimic-cxr":
        if section == "findings":
            experiments["Literature"] = (
                os.path.join(result_dir, "exp-baselines"),
                [
                    (
                        "LaB-RAG",
                        "labrag_findings_METRICS.csv",
                    ),
                    (
                        "RGRG",
                        "rgrg_findings_METRICS.csv",
                    ),
                    (
                        "CheXagent",
                        "chexagent_findings_METRICS.csv",
                    ),
                    (
                        "CXRMate",
                        "cxrmate_findings_METRICS.csv",
                    ),
                ],
            )
        elif section == "impression":
            experiments["Literature"] = (
                os.path.join(result_dir, "exp-baselines"),
                [
                    (
                        "LaB-RAG",
                        "labrag_impression_METRICS.csv",
                    ),
                    (
                        "CXR-RePaiR",
                        "cxrrepair_impression_METRICS.csv",
                    ),
                    (
                        "CXR-ReDonE",
                        "cxrredone_impression_METRICS.csv",
                    ),
                    (
                        "X-REM",
                        "xrem_impression_METRICS.csv",
                    ),
                    (
                        "CheXagent",
                        "chexagent_impression_METRICS.csv",
                    ),
                    (
                        "CXRMate",
                        "cxrmate_impression_METRICS.csv",
                    ),
                ],
            )
    return experiments


COLOR2MODELS = {
    (0.40569574036511175, 0.3832048681541582, 0.8262068965517242): [
        # CheXagent
        "chexagent_findings_METRICS.csv",
        "chexagent_impression_METRICS.csv",
    ],
    (0.7678431372549019, 0.22098039215686274, 0.3531372549019608): [
        # CXRMate
        "cxrmate_findings_METRICS.csv",
        "cxrmate_impression_METRICS.csv",
    ],
    (0.5620270875001143, 0.3477601669452133, 0.8416123820743948): [
        # CXR-ReDonE
        "cxrredone_impression_METRICS.csv",
    ],
    (0.9419607843137255, 0.3950980392156863, 0.06294117647058822): [
        # CXR-RePaiR
        "cxrrepair_impression_METRICS.csv",
    ],
    (0.9, 0.6774509803921569, 0.07098039215686275): [
        # RGRG
        "rgrg_findings_METRICS.csv",
    ],
    (0.9, 0.8805882352941177, 0.44823529411764707): [
        # X-REM
        "xrem_impression_METRICS.csv",
    ],
    (0.5019607843137255, 0.6941176470588235, 0.8274509803921568): [
        # LaB-RAG
        "findings_top-5_mimic-cxr-biovilt-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_mimic-cxr-biovilt-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "findings_top-5_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "labrag_findings_METRICS.csv",
        "labrag_impression_METRICS.csv",
    ],
    (0.5529411764705883, 0.8274509803921568, 0.7803921568627451): [
        # Filter - No-filter
        "findings_top-5_mimic-cxr-biovilt-pred-label_no-filter_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_mimic-cxr-biovilt-pred-label_no-filter_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "findings_top-5_chexpertplus-gloria-pred-label_no-filter_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_chexpertplus-gloria-pred-label_no-filter_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
    ],
    (1.0, 1.0, 0.7019607843137254): [
        # Filter - Partial
        "findings_top-5_mimic-cxr-biovilt-pred-label_partial_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_mimic-cxr-biovilt-pred-label_partial_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "findings_top-5_chexpertplus-gloria-pred-label_partial_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_chexpertplus-gloria-pred-label_partial_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
    ],
    (0.7450980392156863, 0.7294117647058823, 0.8549019607843137): [
        # Prompt - Naive
        "findings_top-5_mimic-cxr-biovilt-pred-label_exact_naive_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_mimic-cxr-biovilt-pred-label_exact_naive_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "findings_top-5_chexpertplus-gloria-pred-label_exact_naive_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_chexpertplus-gloria-pred-label_exact_naive_Mistral-7B-Instruct-v0.3_METRICS.csv",
    ],
    (0.984313725490196, 0.5019607843137255, 0.4470588235294118): [
        # Prompt - Verbose
        "findings_top-5_mimic-cxr-biovilt-pred-label_exact_verbose_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_mimic-cxr-biovilt-pred-label_exact_verbose_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "findings_top-5_chexpertplus-gloria-pred-label_exact_verbose_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_chexpertplus-gloria-pred-label_exact_verbose_Mistral-7B-Instruct-v0.3_METRICS.csv",
    ],
    (0.9921568627450981, 0.7058823529411765, 0.3843137254901961): [
        # Prompt - Instruct
        "findings_top-5_mimic-cxr-biovilt-pred-label_exact_instruct_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_mimic-cxr-biovilt-pred-label_exact_instruct_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "findings_top-5_chexpertplus-gloria-pred-label_exact_instruct_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_chexpertplus-gloria-pred-label_exact_instruct_Mistral-7B-Instruct-v0.3_METRICS.csv",
    ],
    (0.7019607843137254, 0.8705882352941177, 0.4117647058823529): [
        # LLM - Mistral v1
        "findings_top-5_mimic-cxr-biovilt-pred-label_exact_simple_Mistral-7B-Instruct-v0.1_METRICS.csv",
        "impression_top-5_mimic-cxr-biovilt-pred-label_exact_simple_Mistral-7B-Instruct-v0.1_METRICS.csv",
        "findings_top-5_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.1_METRICS.csv",
        "impression_top-5_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.1_METRICS.csv",
    ],
    (0.9882352941176471, 0.803921568627451, 0.8980392156862745): [
        # LLM - BioMistral
        "findings_top-5_mimic-cxr-biovilt-pred-label_exact_simple_BioMistral-7B_METRICS.csv",
        "impression_top-5_mimic-cxr-biovilt-pred-label_exact_simple_BioMistral-7B_METRICS.csv",
        "findings_top-5_chexpertplus-gloria-pred-label_exact_simple_BioMistral-7B_METRICS.csv",
        "impression_top-5_chexpertplus-gloria-pred-label_exact_simple_BioMistral-7B_METRICS.csv",
    ],
    (0.8509803921568627, 0.8509803921568627, 0.8509803921568627): [
        # Label - True
        "findings_top-5_true-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_true-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "findings_top-5_true-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_true-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
    ],
    (0.7372549019607844, 0.5019607843137255, 0.7411764705882353): [
        # Core - No-filter, Naive-prompt
        "findings_top-5_mimic-cxr-biovilt-pred-label_no-filter_naive_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_mimic-cxr-biovilt-pred-label_no-filter_naive_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "findings_top-5_chexpertplus-gloria-pred-label_no-filter_naive_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_chexpertplus-gloria-pred-label_no-filter_naive_Mistral-7B-Instruct-v0.3_METRICS.csv",
    ],
    (0.8, 0.9215686274509803, 0.7725490196078432): [
        # Top-K - 3
        "findings_top-3_mimic-cxr-biovilt-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-3_mimic-cxr-biovilt-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "findings_top-3_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-3_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
    ],
    (1.0, 0.9294117647058824, 0.43529411764705883): [
        # Top-K - 10
        "findings_top-10_mimic-cxr-biovilt-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-10_mimic-cxr-biovilt-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "findings_top-10_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-10_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
    ],
    (0.7686274509803922, 0.611764705882353, 0.5803921568627451): [
        # Embedding - ResNet50
        "findings_top-5_mimic-cxr-resnet50-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_mimic-cxr-resnet50-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "findings_top-5_chexpertplus-resnet50-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_chexpertplus-resnet50-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
    ],
}

MODEL2COLOR = {m: c for c, ms in COLOR2MODELS.items() for m in ms}
