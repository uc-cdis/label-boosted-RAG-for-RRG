# Experiments and Replication of Results

This document serves to detail the experiments we conducted. Precise commands to replicate those experiments are provided in our [run scripts](../scripts/). We additionally provide a utility notebook to generate the run scripts for our experiment set: [`prepare-experiment-scripts.ipynb`](../scripts/prepare-experiment-scripts.ipynb).

### Prerequisites
1. [Create the required python environment](README.md#environment-setup)
1. [Download and preprocess data](https://github.com/StevenSong/cxr-data-ingest)

## Experimental Procedure
The experimental procedure is divided into four parts. Part 1 is extracting features using pretrained vision models. Part 2 is training classifiers over the image features of the train set and applying the classifiers over the test set. This generates labels for use in part 3. Part 3 is testing various strategies for retrieving and prompting an LLM to generate reports over the test set using sample labels. Part 4 is evaluating the generated reports against the true reports.

1. `extract.py` extracts image features using a pretrained image embedding model.
    ```bash
    python /path/to/labrag-repo/rrg/extract.py \
    --model_type biovil-t|gloria|resnet50 \
    --input_path /path/to/dataset/image/files \
    --file_ext .jpg|.png \
    --output_h5 /path/to/rrg-data/features.h5 \
    --num_workers 16 \
    --batch_size 32
    ```
1. `classify.py` trains per-label logistic binary classifiers over the train split, and generates labels over the test split.
    ```bash
    python /path/to/labrag-repo/rrg/classify.py \
    --split_csv /path/to/dataset/split.csv \
    --metadata_csv /path/to/dataset/metadata.csv \
    --label_csv /path/to/cxr-derived/labels.csv \
    --feature_h5 /path/to/rrg-data/features.h5 \
    --output_results /path/to/rrg-data/classifiers
    ```
1. `generate.py` does RAG for radiology reports using configurable LLMs, embedding features, number of retrieved examples, retrieval label filtering, prompt label verbosity and templates, target report section for generation, and inference label set.
    ```bash
    python /path/to/labrag-repo/rrg/generate.py \
    --model hf_model_name \
    --filter_type no-filter|exact|partial \
    --prompt_type naive|simple|verbose|instruct \
    --section_type findings|impression \
    --k K \
    --batch_size 32 \
    --prompt_yaml /path/to/labrag-repo/rrg/prompts.yaml \
    --split_csv /path/to/dataset/split.csv \
    --metadata_csv /path/to/dataset/metadata.csv \
    --true_label_csv /path/to/cxr-derived/labels.csv \
    --predicted_label_csv /path/to/rrg-data/classifiers/pred.csv \
    --report_csv /path/to/dataset/report.csv \
    --feature_h5 /path/to/rrg-data/features.h5 \
    --output_dir /path/to/rrg-data/experiment-output
    ```
1. `eval.py` runs evaluation metrics for radiology report generation.
    ```bash
    python /path/to/labrag-repo/rrg/eval.py \
    --report_csv /path/to/rrg-data/experiment-output/reports.csv \
    --output_csv /path/to/rrg-data/experiment-output/metrics.csv
    ```

## Experimental Trials

**NB**: <u>All experiments are repeated across both Findings and Impression section, and additionally across both MIMIC-CXR and CheXpert Plus datasets</u>. We tie embedding model to dataset so BioViL-T is used as the default image embedding model for MIMIC-CXR and GLoRIA for CheXPert Plus. Additionally, the predicted labels are always from a classifier trained over the specified image embedding model, regardless of what label type is used (i.e. CheXbert vs CheXpert).

The default hyperparameters of LaB-RAG are as follows:
```
Filter: Exact
Format: Simple
Label:  Pred-CheXbert
Embed:  BioViL-T/GLoRIA
LLM:    Mistral-v3
Top-k:  5
```

The set of hyperparameters we investigate is below:
```
Filter: [No-filter, Exact, Partial]
Format: [Naive, Simple, Verbose, Instruct]
Label:  [Pred-CheXbert, Pred-CheXpert, True-CheXbert, True-CheXpert]
Embed:  [BioViL-T/GLoRIA, ResNet50]
LLM:    [Mistral-v1, BioMistral, Mistral-v3]
Top-k:  [3, 5, 10]
```

### Experiments:
1. **Core Experiment**
    <details open>
    <summary>Hyperparameters</summary>

    ```
    Filter: [No-filter, Exact]
    Format: [Naive, Simple]
    Label:  Pred-CheXbert
    Embed:  BioViL-T/GLoRIA
    LLM:    Mistral-v3
    Top-k:  5
    ```
    </details>
1. **Filter Variants**
    <details open>
    <summary>Hyperparameters</summary>

    ```
    Filter: [No-filter, Exact, Partial]
    Format: Simple
    Label:  Pred-CheXbert
    Embed:  BioViL-T/GLoRIA
    LLM:    Mistral-v3
    Top-k:  5
    ```
    </details>
1. **Format Variants**
    <details open>
    <summary>Hyperparameters</summary>

    ```
    Filter: Exact
    Format: [Naive, Simple, Verbose, Instruct]
    Label:  Pred-CheXbert
    Embed:  BioViL-T/GLoRIA
    LLM:    Mistral-v3
    Top-k:  5
    ```
    </details>
1. **Label Quality**
    <details open>
    <summary>Hyperparameters</summary>

    ```
    Filter: Exact
    Format: Simple
    Label:  [Pred-CheXbert, Pred-CheXpert, True-CheXbert, True-CheXpert]
    Embed:  BioViL-T/GLoRIA
    LLM:    Mistral-v3
    Top-k:  5
    ```
    </details>
1. **Image Embedding Model**
    <details open>
    <summary>Hyperparameters</summary>

    ```
    Filter: Exact
    Format: Simple
    Label:  Pred-CheXbert
    Embed:  [BioViL-T/GLoRIA, ResNet50]
    LLM:    Mistral-v3
    Top-k:  5
    ```
    </details>
1. **Language Model**
    <details open>
    <summary>Hyperparameters</summary>

    ```
    Filter: Exact
    Format: Simple
    Label:  Pred-CheXbert
    Embed:  BioViL-T/GLoRIA
    LLM:    [Mistral-v1, BioMistral, Mistral-v3]
    Top-k:  5
    ```
    </details>
1. **Retrieved Samples**
    <details open>
    <summary>Hyperparameters</summary>

    ```
    Filter: Exact
    Format: Simple
    Label:  Pred-CheXbert
    Embed:  BioViL-T/GLoRIA
    LLM:    Mistral-v3
    Top-k:  [3, 5, 10]
    ```
    </details>
