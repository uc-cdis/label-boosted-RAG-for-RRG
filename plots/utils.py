import os
from collections import defaultdict
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator

ALL_METRICS = [
    "bleu4",
    "rougeL",
    "bertscore",
    "radcliqv1",
    "ratescore",
    "f1chexbert",
    "f1radgraph",
    "green",
    "clear_label_presence",
    "clear_severity",
    "clear_descriptive_location",
    "clear_recommendation",
]

METRIC_DISPLAY_NAMES = {
    "bleu4": "BL-4",
    "rougeL": "RG-L",
    "bertscore": "BERT",
    "f1radgraph": "F1-RG",
    "f1chexbert": "F1-CXB",
    "green": "GREEN",
    "ratescore": "RATE",
    "radcliqv1": "1/RCQ",
    "clear_label_presence": "CLR-LP",
    "clear_severity": "CLR-SV",
    "clear_descriptive_location": "CLR-LOC",
    "clear_recommendation": "CLR-REC",
}


def filter_existing_trials(
    *,  # enforce kwargs
    exp_name: str,
    exp_dir: str,
    exp_trials: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    existing_trials = []
    missing_trials = []
    for trial_name, trial_file in exp_trials:
        trial_path = os.path.join(exp_dir, trial_file)
        if os.path.exists(trial_path):
            existing_trials.append((trial_name, trial_file))
        else:
            missing_trials.append((trial_name, trial_path))

    available_names = [trial_name for trial_name, _ in existing_trials]
    missing_names = [trial_name for trial_name, _ in missing_trials]
    print(f"{exp_name}:")
    print(f"  available: {available_names if available_names else 'none'}")
    print(f"  missing: {missing_names if missing_names else 'none'}")
    if missing_trials:
        for _, trial_path in missing_trials:
            print(f"    - {trial_path}")

    return existing_trials


def get_metric_display_names(metrics: list[str]) -> list[str]:
    return [METRIC_DISPLAY_NAMES[m] for m in metrics]


def get_plot_metric_names(metrics: list[str]) -> list[str]:
    return ["1/radcliqv1" if m == "radcliqv1" else m for m in metrics]


def get_metric_display_mean(metric: str, trial_df: pd.DataFrame) -> float:
    if metric == "radcliqv1":
        radcliq_mean = trial_df[metric].mean()
        if radcliq_mean <= 0:
            raise ValueError(
                "Mean RadCliqv1 must be positive to display 1 / mean(RadCliqv1)"
            )
        return 1 / radcliq_mean
    return trial_df[metric].mean()


def check_duplicate_runs(result_dir):
    print("===== Checking duplicate runs are equivalent =====\n")
    compared_groups = 0
    skipped_groups = 0
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

            cols = ALL_METRICS
            dupes = {k: v for k, v in count.items() if len(v) > 1}
            print("Duplicates:\n")
            for k, vs in dupes.items():
                print(k)
                try:
                    group_dfs = []
                    for v in vs:
                        df = pd.read_csv(v)
                        group_dfs.append(df)
                    ref = group_dfs[0]
                    print(f"--- ref: {vs[0]}")
                    for df, v in zip(group_dfs[1:], vs[1:]):
                        print(f"--- cmp: {v}")
                        assert (ref["study_id"] == df["study_id"]).all()
                        assert np.isclose(
                            ref[cols], df[cols], rtol=1e-5, atol=1e-6
                        ).all()
                    compared_groups += 1
                except FileNotFoundError as e:
                    print(f"  WARNING: {e}, skipping")
                    skipped_groups += 1
                print()
    if skipped_groups:
        print(
            "===== All available duplicate runs are equivalent "
            f"({compared_groups} compared, {skipped_groups} skipped due to missing files)! ====="
        )
    else:
        print(
            f"===== All duplicate runs are equivalent ({compared_groups} compared)! ====="
        )


def check_metric_nan_rows(result_dir):
    print("===== Checking metric columns for missing values =====\n")
    metric_files = sorted(Path(result_dir).rglob("*_METRICS.csv"))
    flagged_files = 0
    for path in metric_files:
        df = pd.read_csv(path)
        metric_cols = [col for col in ALL_METRICS if col in df.columns]
        nan_cols = [col for col in metric_cols if df[col].isna().any()]
        if not nan_cols:
            continue

        flagged_files += 1
        print(path)
        for col in nan_cols:
            bad = df[col].isna()
            cols_to_show = [c for c in ["study_id", "dicom_id", col] if c in df.columns]
            print(f"  {col}: {int(bad.sum())} missing")
            print(df.loc[bad, cols_to_show])
            print()

    if flagged_files:
        print(f"===== Found missing metric values in {flagged_files} file(s) =====")
    else:
        print("===== No missing metric values found =====")


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


def get_radcliq_bootstrap_intervals(
    *,
    exp_trials: list[tuple[str, str]],
    trial_dfs: list[pd.DataFrame],
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state: int = 0,
) -> dict[str, tuple[float, float, float]]:
    intervals = {}
    for (trial_name, _), trial_df in zip(exp_trials, trial_dfs):
        values = trial_df["radcliqv1"].to_numpy(dtype=float)
        if values.size == 0:
            raise ValueError("Cannot bootstrap empty values")

        mean_val = values.mean()
        if mean_val <= 0:
            raise ValueError(
                "Mean RadCliqv1 must be positive to display 1 / mean(RadCliqv1)"
            )

        rng = np.random.default_rng(random_state)
        boot_stats = np.empty(n_boot, dtype=float)

        for i in range(n_boot):
            sample = rng.choice(values, size=values.size, replace=True)
            sample_mean = sample.mean()
            if sample_mean <= 0:
                raise ValueError("Bootstrap sample mean RadCliqv1 must stay positive")
            boot_stats[i] = 1 / sample_mean

        alpha = 1 - ci
        ci_low, ci_high = np.percentile(
            boot_stats,
            [100 * alpha / 2, 100 * (1 - alpha / 2)],
        )
        intervals[trial_name] = (1 / mean_val, ci_low, ci_high)
    return intervals


def plot_experiment_bar(
    *,  # enforce kwargs
    title: str,
    exp_name: str,
    exp_trials: list[tuple[str, str]],
    trial_dfs: list[pd.DataFrame],
    metrics: list[str],
) -> plt.Figure:
    lexical_metrics = ["bleu4", "rougeL", "bertscore"]
    semantic_metrics = [
        "f1radgraph",
        "f1chexbert",
        "green",
        "ratescore",
        "radcliqv1",
    ]
    clear_metrics = [
        "clear_label_presence",
        "clear_severity",
        "clear_descriptive_location",
        "clear_recommendation",
    ]
    metric_groups = [
        ([m for m in metrics if m in lexical_metrics], "Lexical"),
        ([m for m in metrics if m in semantic_metrics], "Clinical"),
        ([m for m in metrics if m in clear_metrics], "CLEAR"),
    ]
    metric_groups = [
        (group_metrics, group_label)
        for group_metrics, group_label in metric_groups
        if group_metrics
    ]
    if not metric_groups:
        raise ValueError("No metrics available for plotting")

    # setup dataframe for seaborn barplot
    melted_results = []
    for trial_df, (trial_name, _) in zip(trial_dfs, exp_trials):
        trial_df = trial_df[["study_id"] + metrics].melt(
            id_vars="study_id", var_name="metric"
        )
        trial_df[exp_name] = trial_name
        melted_results.append(trial_df)
    df = pd.concat(melted_results, ignore_index=True)

    # NOTE: Keep raw RadCliqv1 values for paired t-tests to operate on the
    # original per-study scores. The plotted summary uses 1 / mean(RadCliqv1)
    # in plot_experiment_bar for display only.
    df_display = df.copy()
    for trial_name, trial_df in zip(
        hue_order := [trial_name for trial_name, _ in exp_trials], trial_dfs
    ):
        if "radcliqv1" in metrics and "radcliqv1" in trial_df.columns:
            mask = (df_display["metric"] == "radcliqv1") & (
                df_display[exp_name] == trial_name
            )
            df_display.loc[mask, "value"] = get_metric_display_mean(
                "radcliqv1", trial_df
            )

    # setup seaborn barplot parameters
    x = "metric"
    y = "value"
    hue = exp_name
    palette = [MODEL2COLOR[trial_file] for _, trial_file in exp_trials]

    max_cols = max(len(g) for g, _ in metric_groups)
    fig, axes = plt.subplots(
        len(metric_groups),
        1,
        figsize=(max(8, 2.2 * max_cols), 3 * len(metric_groups)),
    )
    axes = np.atleast_1d(axes)

    # Compute RadCliqv1 CI using bootstrap
    radcliq_bootstrap_ci = None
    if "radcliqv1" in metrics:
        radcliq_bootstrap_ci = get_radcliq_bootstrap_intervals(
            exp_trials=exp_trials,
            trial_dfs=trial_dfs,
        )

    for ax, (group_metrics, group_label) in zip(axes, metric_groups):
        order = group_metrics
        group_df = df[df["metric"].isin(group_metrics)]
        group_df_display = df_display[df_display["metric"].isin(group_metrics)]

        if exp_name == "Literature":
            pairs = [
                ((metric, "LaB-RAG"), (metric, n2))
                for metric in group_metrics
                for n2 in hue_order[1:]
            ]
        else:
            pairs = [
                ((metric, n1), (metric, n2))
                for metric in group_metrics
                for i, n1 in enumerate(hue_order)
                for n2 in hue_order[i + 1 :]
            ]

        sns.barplot(
            group_df_display,
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
        if radcliq_bootstrap_ci is not None and "radcliqv1" in group_metrics:
            metric_idx = group_metrics.index("radcliqv1")
            total_width = 0.15 * len(hue_order)
            single_width = total_width / len(hue_order)
            left_edge = metric_idx - total_width / 2
            for hue_idx, trial_name in enumerate(hue_order):
                x_center = left_edge + (hue_idx + 0.5) * single_width
                center, ci_low, ci_high = radcliq_bootstrap_ci[trial_name]
                lower_err = max(0.0, center - ci_low)
                upper_err = max(0.0, ci_high - center)

                ax.errorbar(
                    x_center,
                    center,
                    yerr=[[lower_err], [upper_err]],
                    fmt="none",
                    ecolor="black",
                    elinewidth=1,
                    capsize=0,
                    zorder=30,
                )
        annot = Annotator(
            ax,
            pairs,
            data=group_df,
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
        annot.apply_test().annotate(line_offset=10000)

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim([0.0, 1.0])
        ax.set_xlim([-0.5, len(group_metrics) - 0.5])
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.grid(which="major", axis="y", zorder=0)
        ax.set_title(group_label, fontsize=10)
        ax.set_xticklabels(get_plot_metric_names(order), fontsize=10)
        if ax == axes[0]:
            legend = ax.legend(title=None, loc="upper left")
            legend.set_zorder(10)
        else:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

    fig.suptitle(f"{title}, N={len(trial_dfs[0])}", fontsize=12)
    fig.tight_layout()
    return fig


def plot_experiment_radar(
    *,  # enforce kwargs
    title: str,
    exp_name: str,
    exp_trials: list[tuple[str, str]],
    trial_dfs: list[pd.DataFrame],
    metrics: list[str],
) -> plt.Figure:
    del exp_name  # plotting uses trial names only

    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    raw_means = {
        metric: [get_metric_display_mean(metric, trial_df) for trial_df in trial_dfs]
        for metric in metrics
    }
    metric_min = {metric: 0 for metric in metrics}
    metric_max = {
        metric: (max(values) * 1.1 if max(values) > 0 else 1.0)
        for metric, values in raw_means.items()
    }

    def to_radial(metric: str, value: float) -> float:
        return (value - metric_min[metric]) / (metric_max[metric] - metric_min[metric])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for trial_df, (trial_name, trial_file) in zip(trial_dfs, exp_trials):
        values = [
            to_radial(metric, get_metric_display_mean(metric, trial_df))
            for metric in metrics
        ]
        values += values[:1]
        color = MODEL2COLOR[trial_file]
        ax.plot(angles, values, linewidth=2, label=trial_name, color=color)
        ax.fill(angles, values, alpha=0.05, color=color)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.5])
    ax.set_yticklabels([], fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(get_plot_metric_names(metrics), fontsize=9)

    n_ticks = 3
    for i, metric in enumerate(metrics):
        angle = angles[i]
        for j in range(n_ticks):
            r = j / (n_ticks - 1)
            actual_val = metric_min[metric] + r * (
                metric_max[metric] - metric_min[metric]
            )
            ax.text(
                angle,
                r,
                f"{actual_val:.2f}",
                fontsize=6,
                ha="center",
                va="center",
                color="grey",
                alpha=0.95,
                bbox=dict(
                    boxstyle="round,pad=0.1",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.7,
                ),
            )

    ax.set_title(f"{title}, N={len(trial_dfs[0])}", fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    fig.tight_layout()
    return fig


plot_experiment = plot_experiment_bar


def get_experiments_metadata(
    *,  # enforce kwargs
    dataset: Literal["mimic-cxr", "chexpertplus"],
    section: Literal["findings", "impression"],
    result_dir: str,
):
    if dataset == "mimic-cxr":
        emb_type = "BioViL-T"
        label_type = "biovilt-chexbert-pr-pred"
        alt_label_type = "biovilt-chexpert-pr-pred"
        dataset_dir = "exp-mimic"
    elif dataset == "chexpertplus":
        emb_type = "GLoRIA"
        label_type = "gloria-chexbert-pr-pred"
        alt_label_type = "gloria-chexpert-pr-pred"
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
                    f"{section}_top-5_resnet50-chexbert-pr-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
                ),
            ],
        ),
        "Label Quality": (
            os.path.join(result_dir, dataset_dir, f"exp-{section}", "exp-true-label"),
            [
                (
                    "Extracted - CheXbert",
                    f"{section}_top-5_chexbert-true-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
                ),
                (
                    "Extracted - CheXpert",
                    f"{section}_top-5_chexpert-true-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
                ),
                (
                    "Predicted - CheXbert",
                    f"{section}_top-5_{label_type}-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
                ),
                (
                    "Predicted - CheXpert",
                    f"{section}_top-5_{alt_label_type}-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
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
        "findings_top-5_biovilt-chexbert-pr-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_biovilt-chexbert-pr-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "findings_top-5_gloria-chexbert-pr-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_gloria-chexbert-pr-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "labrag_findings_METRICS.csv",
        "labrag_impression_METRICS.csv",
    ],
    (0.5529411764705883, 0.8274509803921568, 0.7803921568627451): [
        # Filter - No-filter
        "findings_top-5_biovilt-chexbert-pr-pred-label_no-filter_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_biovilt-chexbert-pr-pred-label_no-filter_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "findings_top-5_gloria-chexbert-pr-pred-label_no-filter_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_gloria-chexbert-pr-pred-label_no-filter_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
    ],
    (1.0, 1.0, 0.7019607843137254): [
        # Filter - Partial
        "findings_top-5_biovilt-chexbert-pr-pred-label_partial_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_biovilt-chexbert-pr-pred-label_partial_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "findings_top-5_gloria-chexbert-pr-pred-label_partial_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_gloria-chexbert-pr-pred-label_partial_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
    ],
    (0.7450980392156863, 0.7294117647058823, 0.8549019607843137): [
        # Prompt - Naive
        "findings_top-5_biovilt-chexbert-pr-pred-label_exact_naive_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_biovilt-chexbert-pr-pred-label_exact_naive_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "findings_top-5_gloria-chexbert-pr-pred-label_exact_naive_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_gloria-chexbert-pr-pred-label_exact_naive_Mistral-7B-Instruct-v0.3_METRICS.csv",
    ],
    (0.984313725490196, 0.5019607843137255, 0.4470588235294118): [
        # Prompt - Verbose
        "findings_top-5_biovilt-chexbert-pr-pred-label_exact_verbose_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_biovilt-chexbert-pr-pred-label_exact_verbose_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "findings_top-5_gloria-chexbert-pr-pred-label_exact_verbose_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_gloria-chexbert-pr-pred-label_exact_verbose_Mistral-7B-Instruct-v0.3_METRICS.csv",
    ],
    (0.9921568627450981, 0.7058823529411765, 0.3843137254901961): [
        # Prompt - Instruct
        "findings_top-5_biovilt-chexbert-pr-pred-label_exact_instruct_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_biovilt-chexbert-pr-pred-label_exact_instruct_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "findings_top-5_gloria-chexbert-pr-pred-label_exact_instruct_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_gloria-chexbert-pr-pred-label_exact_instruct_Mistral-7B-Instruct-v0.3_METRICS.csv",
    ],
    (0.7019607843137254, 0.8705882352941177, 0.4117647058823529): [
        # LLM - Mistral v1
        "findings_top-5_biovilt-chexbert-pr-pred-label_exact_simple_Mistral-7B-Instruct-v0.1_METRICS.csv",
        "impression_top-5_biovilt-chexbert-pr-pred-label_exact_simple_Mistral-7B-Instruct-v0.1_METRICS.csv",
        "findings_top-5_gloria-chexbert-pr-pred-label_exact_simple_Mistral-7B-Instruct-v0.1_METRICS.csv",
        "impression_top-5_gloria-chexbert-pr-pred-label_exact_simple_Mistral-7B-Instruct-v0.1_METRICS.csv",
    ],
    (0.9882352941176471, 0.803921568627451, 0.8980392156862745): [
        # LLM - BioMistral
        "findings_top-5_biovilt-chexbert-pr-pred-label_exact_simple_BioMistral-7B_METRICS.csv",
        "impression_top-5_biovilt-chexbert-pr-pred-label_exact_simple_BioMistral-7B_METRICS.csv",
        "findings_top-5_gloria-chexbert-pr-pred-label_exact_simple_BioMistral-7B_METRICS.csv",
        "impression_top-5_gloria-chexbert-pr-pred-label_exact_simple_BioMistral-7B_METRICS.csv",
    ],
    (0.8509803921568627, 0.8509803921568627, 0.8509803921568627): [
        # Label - True - CheXbert
        "findings_top-5_chexbert-true-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_chexbert-true-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "findings_top-5_chexbert-true-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_chexbert-true-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
    ],
    (0.761, 0.761, 0.761): [
        # Label - True - CheXpert
        "findings_top-5_chexpert-true-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_chexpert-true-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "findings_top-5_chexpert-true-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_chexpert-true-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
    ],
    (0.455, 0.62, 0.741): [
        # Label - Predicted - CheXpert
        "findings_top-5_biovilt-chexpert-pr-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_biovilt-chexpert-pr-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "findings_top-5_gloria-chexpert-pr-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_gloria-chexpert-pr-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
    ],
    (0.7372549019607844, 0.5019607843137255, 0.7411764705882353): [
        # Core - No-filter, Naive-prompt
        "findings_top-5_biovilt-chexbert-pr-pred-label_no-filter_naive_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_biovilt-chexbert-pr-pred-label_no-filter_naive_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "findings_top-5_gloria-chexbert-pr-pred-label_no-filter_naive_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_gloria-chexbert-pr-pred-label_no-filter_naive_Mistral-7B-Instruct-v0.3_METRICS.csv",
    ],
    (0.8, 0.9215686274509803, 0.7725490196078432): [
        # Top-K - 3
        "findings_top-3_biovilt-chexbert-pr-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-3_biovilt-chexbert-pr-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "findings_top-3_gloria-chexbert-pr-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-3_gloria-chexbert-pr-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
    ],
    (1.0, 0.9294117647058824, 0.43529411764705883): [
        # Top-K - 10
        "findings_top-10_biovilt-chexbert-pr-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-10_biovilt-chexbert-pr-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "findings_top-10_gloria-chexbert-pr-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-10_gloria-chexbert-pr-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
    ],
    (0.7686274509803922, 0.611764705882353, 0.5803921568627451): [
        # Embedding - ResNet50
        "findings_top-5_resnet50-chexbert-pr-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_resnet50-chexbert-pr-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "findings_top-5_resnet50-chexbert-pr-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
        "impression_top-5_resnet50-chexbert-pr-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv",
    ],
}

MODEL2COLOR = {m: c for c, ms in COLOR2MODELS.items() for m in ms}
