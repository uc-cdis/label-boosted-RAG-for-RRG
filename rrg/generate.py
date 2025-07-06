import argparse
import os
from typing import get_args

import pandas as pd
import yaml
from _data import (
    DEFAULT_DICOM_ID_COL,
    DEFAULT_FINDINGS_COL,
    DEFAULT_IMG_PROJ_KEY,
    DEFAULT_IMPRESSION_COL,
    DEFAULT_LABELS,
    DEFAULT_PATIENT_ID_COL,
    DEFAULT_SPLIT_COL,
    DEFAULT_STUDY_ID_COL,
    DEFAULT_VIEW_COL,
    DEFAULT_VIEW_ORDER,
    get_per_study_data,
    get_split_features,
    get_split_samples,
)
from _prompt import FILTER_TYPE, PROMPT_TYPE, SECTION_TYPE, prepare_prompt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import trange
from vllm import LLM, SamplingParams

DEFAULT_OTHER_COL = "Other"


def generate_radiology_notes(
    *,  # enforce kwargs
    model: str,
    k: int,
    filter_type: FILTER_TYPE,
    prompt_type: PROMPT_TYPE,
    section_type: SECTION_TYPE,
    batch_size: int,
    prompt_yaml: str,
    split_csv: str,
    metadata_csv: str,
    true_label_csv: str,
    predicted_label_csv: str | None,
    report_csv: str,
    feature_h5: str,
    output_dir: str,
    patient_id_col: str = DEFAULT_PATIENT_ID_COL,
    study_id_col: str = DEFAULT_STUDY_ID_COL,
    dicom_id_col: str = DEFAULT_DICOM_ID_COL,
    split_col: str = DEFAULT_SPLIT_COL,
    view_col: str = DEFAULT_VIEW_COL,
    labels: list[str] = DEFAULT_LABELS,
    view_order: list[str] = DEFAULT_VIEW_ORDER,
    findings_col: str = DEFAULT_FINDINGS_COL,
    impression_col: str = DEFAULT_IMPRESSION_COL,
    feature_key: str = DEFAULT_IMG_PROJ_KEY,
    add_other_label: bool = True,
    other_col: str = DEFAULT_OTHER_COL,
):
    os.makedirs(output_dir, exist_ok=True)
    label_type = "true"
    if predicted_label_csv is not None:
        # assumes predicted labels were generated using "feature_h5" embeddings
        label_type = f"{os.path.basename(feature_h5.replace('.h5', ''))}-pred"
    model_name = os.path.basename(model)
    filename = f"{section_type}_top-{k}_{label_type}-label_{filter_type}_{prompt_type}_{model_name}.csv"
    result_csv = os.path.join(output_dir, filename)
    if os.path.exists(result_csv):
        print("File Exists, Exiting")
        return

    # TODO parameterize hardcoded split remapping
    split_remap = {
        "train": "retrieval",
        "validate": "retrieval",
        "test": "inference",
    }

    # Load and merge data relative to true labels
    retrieval_df = get_per_study_data(
        split_csv=split_csv,
        metadata_csv=metadata_csv,
        label_csv=true_label_csv,
        report_csv=report_csv,
        patient_id_col=patient_id_col,
        study_id_col=study_id_col,
        dicom_id_col=dicom_id_col,
        split_col=split_col,
        view_col=view_col,
        labels=labels,
        view_order=view_order,
        report_cols=[findings_col, impression_col],
        split_remap=split_remap,
    )

    # Load and merge data relative to predicted labels if provided
    inference_df = get_per_study_data(
        split_csv=split_csv,
        metadata_csv=metadata_csv,
        label_csv=predicted_label_csv or true_label_csv,
        report_csv=report_csv,
        patient_id_col=patient_id_col,
        study_id_col=study_id_col,
        dicom_id_col=dicom_id_col,
        split_col=split_col,
        view_col=view_col,
        labels=labels,
        view_order=view_order,
        report_cols=[findings_col, impression_col],
        split_remap=split_remap,
    )

    # Check that true and predicted labels result in same merged dataframes
    cols = [patient_id_col, study_id_col, dicom_id_col, split_col, view_col]
    assert retrieval_df[cols].equals(inference_df[cols])

    # Filter dataset to only those with given section type
    if section_type == "findings":
        report_cols = [findings_col]
    elif section_type == "impression":
        report_cols = [impression_col]
    elif section_type in ["both", "findings-intersect", "impression-intersect"]:
        report_cols = [findings_col, impression_col]
    else:
        raise ValueError(f"Unknown section type: {section_type}")

    mask = retrieval_df[report_cols].notna().all(axis=1)
    retrieval_df = retrieval_df[mask].reset_index(drop=True).copy()
    inference_df = inference_df[mask].reset_index(drop=True).copy()

    # Add implicit "other" label
    if add_other_label:
        # TODO does "other" definition depend on prompt type?
        retrieval_df[other_col] = (retrieval_df[labels] != 1).all(axis=1).astype(int)
        inference_df[other_col] = (inference_df[labels] != 1).all(axis=1).astype(int)
        labels += [other_col]

    # Prepare per-split projected embeddings
    features = get_split_features(
        feature_h5=feature_h5,
        feature_key=feature_key,
        sample_df=retrieval_df,
        patient_id_col=patient_id_col,
        study_id_col=study_id_col,
        dicom_id_col=dicom_id_col,
        split_col=split_col,
    )
    retrieval_features = features["retrieval"]
    inference_features = features["inference"]

    # Prepare per-split metadata, labels, and reports
    retrieval_samples = get_split_samples(
        sample_df=retrieval_df,
        split_col=split_col,
    )["retrieval"]
    inference_samples = get_split_samples(
        sample_df=inference_df,
        split_col=split_col,
    )["inference"]

    # Prepare prompt templates
    with open(prompt_yaml) as f:
        prompt_templates = yaml.safe_load(f)

    # Compute similarity between inference and retrieval samples
    similarity = cosine_similarity(inference_features, retrieval_features)

    # Setup LLM inference engine
    # TODO parameterize decoding strategy
    sampling_params = SamplingParams(  # greedy
        n=1,
        temperature=0,
        use_beam_search=False,
        seed=42,
        max_tokens=512,
    )
    # TODO paramterize vLLM configuration
    llm = LLM(
        model=model,
        trust_remote_code=True,
        enforce_eager=True,
    )

    # Generate reports
    N = len(inference_samples)
    for lo in trange(0, N, batch_size, desc="Batched Generation"):
        hi = lo + batch_size
        if hi > N:
            hi = N

        batch_ids = []
        batch_prompts = []
        batch_targets = []
        batch_retrieved = []
        for i in range(lo, hi):
            batch_ids.append(inference_samples.iloc[i][study_id_col])
            prompt, target_report, retrieved_studies = prepare_prompt(
                retrieval_samples=retrieval_samples,
                target_sample=inference_samples.iloc[i],
                target_similarity=similarity[i],
                k=k,
                prompt_templates=prompt_templates,
                filter_type=filter_type,
                prompt_type=prompt_type,
                section_type=section_type,
                labels=labels,
                findings_col=findings_col,
                impression_col=impression_col,
                study_id_col=study_id_col,
            )
            batch_prompts.append(prompt)
            batch_targets.append(target_report)
            batch_retrieved.append(retrieved_studies)

        batch_outputs = llm.generate(batch_prompts, sampling_params)
        batch_outputs = [o.outputs[0].text for o in batch_outputs]

        # Incrementally save results
        pd.DataFrame(
            {
                study_id_col: batch_ids,
                "retrieved_studies": batch_retrieved,
                "prompt": batch_prompts,
                "actual_text": batch_targets,
                "generated_text": batch_outputs,
            }
        ).to_csv(
            result_csv,
            mode="w" if lo == 0 else "a",
            header=lo == 0,
            index=False,
        )


def optional_empty_str(x):
    return None if x in ["", "None", "none"] else x


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Path to local model or huggingface model name",
    )
    parser.add_argument(
        "--k",
        required=True,
        type=int,
        help="Maximum number of retrieval samples to use for prompt augmentation",
    )
    parser.add_argument(
        "--filter_type",
        required=True,
        choices=get_args(FILTER_TYPE),
        help="Type of filtering strategy for retrieval",
    )
    parser.add_argument(
        "--prompt_type",
        required=True,
        choices=get_args(PROMPT_TYPE),
        help="Type of prompting strategy for generation",
    )
    parser.add_argument(
        "--section_type",
        required=True,
        choices=get_args(SECTION_TYPE),
        help="Section to generate",
    )
    parser.add_argument(
        "--batch_size",
        required=True,
        type=int,
        help="Number of samples for batched inference",
    )
    parser.add_argument(
        "--prompt_yaml",
        required=True,
        help="Path to input prompt store YAML",
    )
    parser.add_argument(
        "--split_csv",
        required=True,
        help="Path to input split csv with patient ID, study ID, dicom ID, and split columns",
    )
    parser.add_argument(
        "--metadata_csv",
        required=True,
        help="Path to input metadata csv with patient ID, study ID, dicom ID, and view position columns",
    )
    parser.add_argument(
        "--true_label_csv",
        required=True,
        help="Path to input ground-truth label csv with study ID and label columns",
    )
    parser.add_argument(
        "--predicted_label_csv",
        required=False,
        type=optional_empty_str,
        help="Path to input predicted label csv with study ID and label columns (exclude to infer using true labels)",
    )
    parser.add_argument(
        "--report_csv",
        required=True,
        help="Path to input report csv with study ID and report columns",
    )
    parser.add_argument(
        "--feature_h5",
        required=True,
        help="Path to input feature h5",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save output generation results",
    )
    parser.add_argument(
        "--patient_id_col",
        default=DEFAULT_PATIENT_ID_COL,
        help="Name of patient ID column in CSV files",
    )
    parser.add_argument(
        "--study_id_col",
        default=DEFAULT_STUDY_ID_COL,
        help="Name of study ID column in CSV files",
    )
    parser.add_argument(
        "--dicom_id_col",
        default=DEFAULT_DICOM_ID_COL,
        help="Name of dicom ID column in CSV files",
    )
    parser.add_argument(
        "--split_col",
        default=DEFAULT_SPLIT_COL,
        help="Name of split column in CSV files",
    )
    parser.add_argument(
        "--view_col",
        default=DEFAULT_VIEW_COL,
        help="Name of view column in CSV files",
    )
    parser.add_argument(
        "--findings_col",
        default=DEFAULT_FINDINGS_COL,
        help="Name of findings column in CSV files",
    )
    parser.add_argument(
        "--impression_col",
        default=DEFAULT_IMPRESSION_COL,
        help="Name of impression column in CSV files",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=DEFAULT_LABELS,
        help="Name of labels",
    )
    parser.add_argument(
        "--view_order",
        nargs="+",
        default=DEFAULT_VIEW_ORDER,
        help="Order of views to use",
    )
    parser.add_argument(
        "--feature_key",
        default=DEFAULT_IMG_PROJ_KEY,
        help="Name of image features",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    generate_radiology_notes(
        model=args.model,
        k=args.k,
        filter_type=args.filter_type,
        prompt_type=args.prompt_type,
        section_type=args.section_type,
        batch_size=args.batch_size,
        prompt_yaml=args.prompt_yaml,
        split_csv=args.split_csv,
        metadata_csv=args.metadata_csv,
        true_label_csv=args.true_label_csv,
        predicted_label_csv=args.predicted_label_csv,
        report_csv=args.report_csv,
        feature_h5=args.feature_h5,
        output_dir=args.output_dir,
        patient_id_col=args.patient_id_col,
        study_id_col=args.study_id_col,
        dicom_id_col=args.dicom_id_col,
        split_col=args.split_col,
        view_col=args.view_col,
        labels=args.labels,
        view_order=args.view_order,
        findings_col=args.findings_col,
        impression_col=args.impression_col,
        feature_key=args.feature_key,
    )
