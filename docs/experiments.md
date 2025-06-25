# Experiments and Replication of Results

## Environment and Data Setup
1. [Create the required python environment](README.md#environment-setup)
1. [Download and preprocess data](data-ingest.md)

## Experimental Setup
The experiments are divided into four parts. Part 1 is extracting features using a pretrained vision model. Part 2 is training a classifier over the image features of the train set and applying the classifier over the test set. This generates labels for use in part 3. Part 3 is testing various strategies for retrieving and prompting an LLM to generate reports over the test set using sample labels. Part 4 is evaluating the generated reports against the true reports.
* Activate the conda environment:
    ```bash
    conda activate labrag
    ```
1. `extract.py` extracts image features using BioViL-T [1].
    ```bash
    python /path/to/labrag-repo/rrg/extract.py \
    --input_path /path/to/mimic-cxr/files \
    --file_ext ".jpg" \
    --output_h5 /path/to/rrg-data/biovilt-features.h5 \
    --num_workers 16 \
    --batch_size 32
    ```
1. `classify.py` trains per-label logistic binary classifiers over the train split, and generates labels over the test split.
    ```bash
    python /path/to/labrag-repo/rrg/classify.py \
    --split_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-split.csv \
    --metadata_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-metadata.csv \
    --label_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv \
    --feature_h5 /path/to/rrg-data/biovilt-features.h5 \
    --output_results /path/to/rrg-data/image-labels
    ```
1. `generate.py` does RAG for radiology reports using configurable LLMs, number of retrieved examples, retrieval label filtering, prompt label verbosity and templates, target section for generation, and inference label set. The exact runs and their command line arguments are detailed in the following section.
1. `eval.py` runs evaluation metrics for radiology report generation.
    ```bash
    python /path/to/labrag-repo/rrg/eval.py \
    --report_csv /path/to/rrg-data/experiment-output.csv \
    --output_csv /path/to/rrg-data/experiment-output-metrics.csv
    ```

### Experiment Runs
1. **Simplest Label Filter & Format**: Investigating the effects of the exact label filter and the simple label format over standard RAG.
    <details open>
    <summary>Hyperparameters</summary>

    ```
    Model:   Mistral v3
    K:       5
    Filter:  [No-filter|Exact]
    Prompt:  [Naive|Simple]
    Section: Findings|Impression
    Labels:  Predicted
    ```
    </details>
    <details>
    <summary>Standard RAG</summary>

    ```bash
    python /path/to/labrag-repo/rrg/generate.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --k 5 \
    --filter_type no-filter \
    --prompt_type naive \
    --section_type findings|impression \
    --batch_size 32 \
    --prompt_yaml /path/to/labrag-repo/rrg/prompts.yaml \
    --split_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-split.csv \
    --metadata_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-metadata.csv \
    --true_label_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv \
    --predicted_label_csv /path/to/rrg-data/image-labels/pred_pr.csv \
    --report_csv /path/to/mimic-cxr/mimic_cxr_sectioned.csv \
    --feature_h5 /path/to/rrg-data/biovilt-features.h5 \
    --output_dir /path/to/rrg-data/exp-boost
    ```
    </details>
    <details>
    <summary>Exact Label Filter only</summary>

    ```bash
    python /path/to/labrag-repo/rrg/generate.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --k 5 \
    --filter_type exact \
    --prompt_type naive \
    --section_type findings|impression \
    --batch_size 32 \
    --prompt_yaml /path/to/labrag-repo/rrg/prompts.yaml \
    --split_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-split.csv \
    --metadata_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-metadata.csv \
    --true_label_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv \
    --predicted_label_csv /path/to/rrg-data/image-labels/pred_pr.csv \
    --report_csv /path/to/mimic-cxr/mimic_cxr_sectioned.csv \
    --feature_h5 /path/to/rrg-data/biovilt-features.h5 \
    --output_dir /path/to/rrg-data/exp-boost
    ```
    </details>
    <details>
    <summary>Simple Label Format only</summary>

    ```bash
    python /path/to/labrag-repo/rrg/generate.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --k 5 \
    --filter_type no-filter \
    --prompt_type simple \
    --section_type findings|impression \
    --batch_size 32 \
    --prompt_yaml /path/to/labrag-repo/rrg/prompts.yaml \
    --split_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-split.csv \
    --metadata_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-metadata.csv \
    --true_label_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv \
    --predicted_label_csv /path/to/rrg-data/image-labels/pred_pr.csv \
    --report_csv /path/to/mimic-cxr/mimic_cxr_sectioned.csv \
    --feature_h5 /path/to/rrg-data/biovilt-features.h5 \
    --output_dir /path/to/rrg-data/exp-boost
    ```
    </details>
    <details>
    <summary>LaB-RAG</summary>

    ```bash
    python /path/to/labrag-repo/rrg/generate.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --k 5 \
    --filter_type exact \
    --prompt_type simple \
    --section_type findings|impression \
    --batch_size 32 \
    --prompt_yaml /path/to/labrag-repo/rrg/prompts.yaml \
    --split_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-split.csv \
    --metadata_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-metadata.csv \
    --true_label_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv \
    --predicted_label_csv /path/to/rrg-data/image-labels/pred_pr.csv \
    --report_csv /path/to/mimic-cxr/mimic_cxr_sectioned.csv \
    --feature_h5 /path/to/rrg-data/biovilt-features.h5 \
    --output_dir /path/to/rrg-data/exp-boost
    ```
    </details>

1. **Label Filtering**: The strategy by which to filter retrieved contexts based on their labels, in addition to baseline vector similarity.
    <details open>
    <summary>Hyperparameters</summary>

    ```
    Model:   Mistral v3
    K:       5
    Filter:  [experiment]
    Prompt:  Simple
    Section: Findings|Impression
    Labels:  Predicted
    ```
    </details>
    <details>
    <summary>No Filtering</summary>

    ```bash
    python /path/to/labrag-repo/rrg/generate.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --k 5 \
    --filter_type no-filter \
    --prompt_type simple \
    --section_type findings|impression \
    --batch_size 32 \
    --prompt_yaml /path/to/labrag-repo/rrg/prompts.yaml \
    --split_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-split.csv \
    --metadata_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-metadata.csv \
    --true_label_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv \
    --predicted_label_csv /path/to/rrg-data/image-labels/pred_pr.csv \
    --report_csv /path/to/mimic-cxr/mimic_cxr_sectioned.csv \
    --feature_h5 /path/to/rrg-data/biovilt-features.h5 \
    --output_dir /path/to/rrg-data/exp-filter
    ```
    </details>
    <details>
    <summary>Exact Filtering</summary>

    ```bash
    python /path/to/labrag-repo/rrg/generate.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --k 5 \
    --filter_type exact \
    --prompt_type simple \
    --section_type findings|impression \
    --batch_size 32 \
    --prompt_yaml /path/to/labrag-repo/rrg/prompts.yaml \
    --split_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-split.csv \
    --metadata_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-metadata.csv \
    --true_label_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv \
    --predicted_label_csv /path/to/rrg-data/image-labels/pred_pr.csv \
    --report_csv /path/to/mimic-cxr/mimic_cxr_sectioned.csv \
    --feature_h5 /path/to/rrg-data/biovilt-features.h5 \
    --output_dir /path/to/rrg-data/exp-filter
    ```
    </details>
    <details>
    <summary>Partial Filtering</summary>

    ```bash
    python /path/to/labrag-repo/rrg/generate.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --k 5 \
    --filter_type partial \
    --prompt_type simple \
    --section_type findings|impression \
    --batch_size 32 \
    --prompt_yaml /path/to/labrag-repo/rrg/prompts.yaml \
    --split_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-split.csv \
    --metadata_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-metadata.csv \
    --true_label_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv \
    --predicted_label_csv /path/to/rrg-data/image-labels/pred_pr.csv \
    --report_csv /path/to/mimic-cxr/mimic_cxr_sectioned.csv \
    --feature_h5 /path/to/rrg-data/biovilt-features.h5 \
    --output_dir /path/to/rrg-data/exp-filter
    ```
    </details>

1. **Label Format & Prompt**: The verbosity of the prompt, including whether we include labels and instructions on how to use those labels.
    <details open>
    <summary>Hyperparameters</summary>

    ```
    Model:   Mistral v3
    K:       5
    Filter:  Exact
    Prompt:  [experiment]
    Section: Findings|Impression
    Labels:  Predicted
    ```
    </details>
    <details>
    <summary>Naive</summary>

    ```bash
    python /path/to/labrag-repo/rrg/generate.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --k 5 \
    --filter_type exact \
    --prompt_type naive \
    --section_type findings|impression \
    --batch_size 32 \
    --prompt_yaml /path/to/labrag-repo/rrg/prompts.yaml \
    --split_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-split.csv \
    --metadata_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-metadata.csv \
    --true_label_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv \
    --predicted_label_csv /path/to/rrg-data/image-labels/pred_pr.csv \
    --report_csv /path/to/mimic-cxr/mimic_cxr_sectioned.csv \
    --feature_h5 /path/to/rrg-data/biovilt-features.h5 \
    --output_dir /path/to/rrg-data/exp-prompt
    ```
    </details>
    <details>
    <summary>Simple</summary>

    ```bash
    python /path/to/labrag-repo/rrg/generate.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --k 5 \
    --filter_type exact \
    --prompt_type simple \
    --section_type findings|impression \
    --batch_size 32 \
    --prompt_yaml /path/to/labrag-repo/rrg/prompts.yaml \
    --split_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-split.csv \
    --metadata_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-metadata.csv \
    --true_label_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv \
    --predicted_label_csv /path/to/rrg-data/image-labels/pred_pr.csv \
    --report_csv /path/to/mimic-cxr/mimic_cxr_sectioned.csv \
    --feature_h5 /path/to/rrg-data/biovilt-features.h5 \
    --output_dir /path/to/rrg-data/exp-prompt
    ```
    </details>
    <details>
    <summary>Verbose</summary>

    ```bash
    python /path/to/labrag-repo/rrg/generate.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --k 5 \
    --filter_type exact \
    --prompt_type verbose \
    --section_type findings|impression \
    --batch_size 32 \
    --prompt_yaml /path/to/labrag-repo/rrg/prompts.yaml \
    --split_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-split.csv \
    --metadata_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-metadata.csv \
    --true_label_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv \
    --predicted_label_csv /path/to/rrg-data/image-labels/pred_pr.csv \
    --report_csv /path/to/mimic-cxr/mimic_cxr_sectioned.csv \
    --feature_h5 /path/to/rrg-data/biovilt-features.h5 \
    --output_dir /path/to/rrg-data/exp-prompt
    ```
    </details>
    <details>
    <summary>Instruct</summary>

    ```bash
    python /path/to/labrag-repo/rrg/generate.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --k 5 \
    --filter_type exact \
    --prompt_type instruct \
    --section_type findings|impression \
    --batch_size 32 \
    --prompt_yaml /path/to/labrag-repo/rrg/prompts.yaml \
    --split_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-split.csv \
    --metadata_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-metadata.csv \
    --true_label_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv \
    --predicted_label_csv /path/to/rrg-data/image-labels/pred_pr.csv \
    --report_csv /path/to/mimic-cxr/mimic_cxr_sectioned.csv \
    --feature_h5 /path/to/rrg-data/biovilt-features.h5 \
    --output_dir /path/to/rrg-data/exp-prompt
    ```
    </details>
1. **Model**: The LLM we use to generate.
    <details open>
    <summary>Hyperparameters</summary>

    ```
    Model:   [experiment]
    K:       5
    Filter:  Exact
    Prompt:  Simple
    Section: Findings|Impression
    Labels:  Predicted
    ```
    </details>
    <details>
    <summary>BioMistral</summary>

    ```bash
    python /path/to/labrag-repo/rrg/generate.py \
    --model BioMistral/BioMistral-7B \
    --k 5 \
    --filter_type exact \
    --prompt_type simple \
    --section_type findings|impression \
    --batch_size 32 \
    --prompt_yaml /path/to/labrag-repo/rrg/prompts.yaml \
    --split_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-split.csv \
    --metadata_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-metadata.csv \
    --true_label_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv \
    --predicted_label_csv /path/to/rrg-data/image-labels/pred_pr.csv \
    --report_csv /path/to/mimic-cxr/mimic_cxr_sectioned.csv \
    --feature_h5 /path/to/rrg-data/biovilt-features.h5 \
    --output_dir /path/to/rrg-data/exp-model
    ```
    </details>
    <details>
    <summary>Mistral v1</summary>

    ```bash
    python /path/to/labrag-repo/rrg/generate.py \
    --model mistralai/Mistral-7B-Instruct-v0.1 \
    --k 5 \
    --filter_type exact \
    --prompt_type simple \
    --section_type findings|impression \
    --batch_size 32 \
    --prompt_yaml /path/to/labrag-repo/rrg/prompts.yaml \
    --split_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-split.csv \
    --metadata_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-metadata.csv \
    --true_label_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv \
    --predicted_label_csv /path/to/rrg-data/image-labels/pred_pr.csv \
    --report_csv /path/to/mimic-cxr/mimic_cxr_sectioned.csv \
    --feature_h5 /path/to/rrg-data/biovilt-features.h5 \
    --output_dir /path/to/rrg-data/exp-model
    ```
    </details>
    <details>
    <summary>Mistral v3</summary>

    ```bash
    python /path/to/labrag-repo/rrg/generate.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --k 5 \
    --filter_type exact \
    --prompt_type simple \
    --section_type findings|impression \
    --batch_size 32 \
    --prompt_yaml /path/to/labrag-repo/rrg/prompts.yaml \
    --split_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-split.csv \
    --metadata_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-metadata.csv \
    --true_label_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv \
    --predicted_label_csv /path/to/rrg-data/image-labels/pred_pr.csv \
    --report_csv /path/to/mimic-cxr/mimic_cxr_sectioned.csv \
    --feature_h5 /path/to/rrg-data/biovilt-features.h5 \
    --output_dir /path/to/rrg-data/exp-model
    ```
    </details>
1. **Labels**: Predicted or true labels at inference.
    <details open>
    <summary>Hyperparameters</summary>

    ```
    Model:   Mistral v3
    K:       5
    Filter:  Exact
    Prompt:  Simple
    Section: Findings|Impression
    Labels:  [experiment]
    ```
    </details>
    <details>
    <summary>Predicted</summary>

    ```bash
    python /path/to/labrag-repo/rrg/generate.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --k 5 \
    --filter_type exact \
    --prompt_type simple \
    --section_type findings|impression \
    --batch_size 32 \
    --prompt_yaml /path/to/labrag-repo/rrg/prompts.yaml \
    --split_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-split.csv \
    --metadata_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-metadata.csv \
    --true_label_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv \
    --predicted_label_csv /path/to/rrg-data/image-labels/pred_pr.csv \
    --report_csv /path/to/mimic-cxr/mimic_cxr_sectioned.csv \
    --feature_h5 /path/to/rrg-data/biovilt-features.h5 \
    --output_dir /path/to/rrg-data/exp-labels
    ```
    </details>
    <details>
    <summary>True</summary>

    ```bash
    # Omit --predicted_label_csv
    python /path/to/labrag-repo/rrg/generate.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --k 5 \
    --filter_type exact \
    --prompt_type simple \
    --section_type findings|impression \
    --batch_size 32 \
    --prompt_yaml /path/to/labrag-repo/rrg/prompts.yaml \
    --split_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-split.csv \
    --metadata_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-metadata.csv \
    --true_label_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv \
    --report_csv /path/to/mimic-cxr/mimic_cxr_sectioned.csv \
    --feature_h5 /path/to/rrg-data/biovilt-features.h5 \
    --output_dir /path/to/rrg-data/exp-labels
    ```
    </details>
1. **Section**: Radiology report section to generate.
    <details open>
    <summary>Hyperparameters</summary>

    ```
    Model:   Mistral v3
    K:       5
    Filter:  Exact
    Prompt:  Simple
    Section: [experiment]
    Labels:  Predicted
    ```
    </details>
    <details>
    <summary>Findings</summary>

    ```bash
    python /path/to/labrag-repo/rrg/generate.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --k 5 \
    --filter_type exact \
    --prompt_type simple \
    --section_type findings-intersect \
    --batch_size 32 \
    --prompt_yaml /path/to/labrag-repo/rrg/prompts.yaml \
    --split_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-split.csv \
    --metadata_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-metadata.csv \
    --true_label_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv \
    --predicted_label_csv /path/to/rrg-data/image-labels/pred_pr.csv \
    --report_csv /path/to/mimic-cxr/mimic_cxr_sectioned.csv \
    --feature_h5 /path/to/rrg-data/biovilt-features.h5 \
    --output_dir /path/to/rrg-data/exp-section
    ```
    </details>
    <details>
    <summary>Impression</summary>

    ```bash
    python /path/to/labrag-repo/rrg/generate.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --k 5 \
    --filter_type exact \
    --prompt_type simple \
    --section_type impression-intersect \
    --batch_size 32 \
    --prompt_yaml /path/to/labrag-repo/rrg/prompts.yaml \
    --split_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-split.csv \
    --metadata_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-metadata.csv \
    --true_label_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv \
    --predicted_label_csv /path/to/rrg-data/image-labels/pred_pr.csv \
    --report_csv /path/to/mimic-cxr/mimic_cxr_sectioned.csv \
    --feature_h5 /path/to/rrg-data/biovilt-features.h5 \
    --output_dir /path/to/rrg-data/exp-section
    ```
    </details>
    <details>
    <summary>Both</summary>

    ```bash
    python /path/to/labrag-repo/rrg/generate.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --k 5 \
    --filter_type exact \
    --prompt_type simple \
    --section_type both \
    --batch_size 32 \
    --prompt_yaml /path/to/labrag-repo/rrg/prompts.yaml \
    --split_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-split.csv \
    --metadata_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-metadata.csv \
    --true_label_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv \
    --predicted_label_csv /path/to/rrg-data/image-labels/pred_pr.csv \
    --report_csv /path/to/mimic-cxr/mimic_cxr_sectioned.csv \
    --feature_h5 /path/to/rrg-data/biovilt-features.h5 \
    --output_dir /path/to/rrg-data/exp-section
    ```
    </details>

## References
1. https://huggingface.co/microsoft/BiomedVLP-BioViL-T
