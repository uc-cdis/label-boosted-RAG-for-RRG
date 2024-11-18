from typing import Literal

import numpy as np
import pandas as pd
from _data import (
    DEFAULT_FINDINGS_COL,
    DEFAULT_IMPRESSION_COL,
    DEFAULT_LABELS,
    DEFAULT_STUDY_ID_COL,
)

FILTER_TYPE = Literal["no-filter", "exact", "partial"]
PROMPT_TYPE = Literal["naive", "simple", "verbose", "instruct"]
SECTION_TYPE = Literal[
    "findings",
    "impression",
    "both",
    "findings-intersect",
    "impression-intersect",
]


cached = None


def prepare_prompt(
    *,  # enforce kwargs
    retrieval_samples: pd.DataFrame,
    target_sample: pd.Series,
    target_similarity: np.ndarray,
    k: int,
    prompt_templates: dict[str, str],
    filter_type: FILTER_TYPE,
    prompt_type: PROMPT_TYPE,
    section_type: SECTION_TYPE,
    labels: list[str] = DEFAULT_LABELS,
    findings_col: str = DEFAULT_FINDINGS_COL,
    impression_col: str = DEFAULT_IMPRESSION_COL,
    study_id_col: str = DEFAULT_STUDY_ID_COL,
    return_relative_idxs: bool = False,
) -> tuple[str, str, str] | tuple[str, str, str, list[int]]:
    target_positives = target_sample[labels].to_numpy()
    target_positives = (target_positives == 1).astype(int)
    global cached
    if cached is None:
        temp = retrieval_samples[labels].to_numpy()
        # TODO parameterize hardcoded positive value
        cached = (temp == 1).astype(int)
    retrieval_positives = cached

    # First sort by similarity metric
    sim_sort_idxs = target_similarity.argsort()

    rel_target_ret_pos = retrieval_positives[sim_sort_idxs]

    target_positives_view = np.broadcast_to(target_positives, rel_target_ret_pos.shape)

    # The below strategies are ambiguous for samples with no positive labels
    # hence why we add an implicit "other" label to handle such cases
    if filter_type == "exact":
        # Number of overlapping labels must exactly match
        mask = (target_positives_view == rel_target_ret_pos).all(axis=1)
        mask = np.argwhere(mask).squeeze(1)
    elif filter_type == "partial":
        # Find number of overlapping labels
        pos_overlap_bits = target_positives_view & rel_target_ret_pos
        num_pos_overlap = pos_overlap_bits.sum(axis=1)

        # Sort by number of overlapping labels
        # Does NOT consider _which_ labels overlap
        mask = num_pos_overlap.argsort(kind="stable")
    elif filter_type == "no-filter":
        # Dummy mask
        mask = np.arange(len(target_similarity))
    else:
        raise ValueError("Unknown filter type: {filter_type}")

    # Retrieve most relevant reference samples
    # May retrieve less than k if using exact match filtering
    k_mask = mask[-k:][::-1]

    # Iterative multi-key index sort requires mapping back to the source indices
    k_idxs = sim_sort_idxs[k_mask]
    k_references = retrieval_samples.iloc[k_idxs]

    # Prepare prompt options
    if prompt_type == "naive":
        make_label = make_naive_label
    elif prompt_type == "simple":
        make_label = make_simple_label
    elif prompt_type in ["verbose", "instruct"]:
        make_label = make_verbose_label
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    prompt_template = prompt_templates[prompt_type]

    section_headers = False
    if section_type == "findings" or section_type == "findings-intersect":
        use_findings = True
        use_impression = False
    elif section_type == "impression" or section_type == "impression-intersect":
        use_findings = False
        use_impression = True
    elif section_type == "both":
        use_findings = True
        use_impression = True
        section_headers = True
    else:
        raise ValueError(f"Unknown section type: {section_type}")

    # Make examples with the below format, depending on prompt and section:
    """
    Example: N
    Labels: X, Y, Z    # if simple
    Positive: X, Y, Z  # if verbose or instruct
    Negative: A, B, C  # if verbose or instruct
    Uncertain: D, E    # if verbose or instruct
    Unmentioned: V, W  # if verbose or instruct
    [Findings: ]Foo    # if findings (w/ header if both)
    [Impression: ]Bar  # if impression (w/ header if both)
    """
    examples = []
    retrieved_studies = []
    for i, (_, reference) in enumerate(k_references.iterrows()):
        example = f"Example: {i+1}\n"
        example += make_label(reference, labels)
        if use_findings:
            if section_headers:
                example += "Findings: "
            example += reference[findings_col] + "\n"
        if use_impression:
            if section_headers:
                example += "Impression: "
            example += reference[impression_col] + "\n"
        examples.append(example)
        retrieved_studies.append(str(reference[study_id_col]))
    retrieved_studies = ", ".join(retrieved_studies)

    # Prepare final prompt and target report
    context = "\n".join(examples)
    target_label = make_label(target_sample, labels)
    # .format ignores extra target_label arg for naive prompt
    prompt = prompt_template.format(context, target_label)

    target_report = ""
    if use_findings:
        if section_headers:
            target_report += "Findings: "
        target_report += target_sample[findings_col] + "\n"
    if use_impression:
        if section_headers:
            target_report += "Impression: "
        target_report += target_sample[impression_col] + "\n"

    if return_relative_idxs:
        return prompt, target_report, retrieved_studies, len(retrieval_samples) - k_mask
    return prompt, target_report, retrieved_studies


def get_labels_of_value(label_series: pd.Series, value: float) -> list[str]:
    if np.isnan(value):
        _labels = label_series.isna()
    else:
        _labels = label_series == value
    return [l for l, b in _labels.items() if b]


def make_naive_label(sample: pd.Series, labels: list[str]) -> str:
    return ""


def make_simple_label(sample: pd.Series, labels: list[str]) -> str:
    # TODO parameterize hardcoded positive value
    pos_labels = get_labels_of_value(sample[labels], 1)
    return "Label: " + ", ".join(pos_labels) + "\n"


def make_verbose_label(sample: pd.Series, labels: list[str]) -> str:
    label_series = sample[labels]
    # TODO parameterize hardcoded pos/neg/uncertain/unmentioned values
    pos_labels = get_labels_of_value(label_series, 1)
    neg_labels = get_labels_of_value(label_series, 0)
    unc_labels = get_labels_of_value(label_series, -1)
    non_labels = get_labels_of_value(label_series, np.nan)
    # fmt: off
    return (
        "Positive: " + ", ".join(pos_labels) + "\n"
        + "Negative: " + ", ".join(neg_labels) + "\n"
        + "Uncertain: " + ", ".join(unc_labels) + "\n"
        + "Unmentioned: " + ", ".join(non_labels) + "\n"
    )
    # fmt: on
