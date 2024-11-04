# Label Boosted RAG for RRG
Experiments around radiology report generation using agentic retrieval augmented generation systems.

### Environment setup
1. [install miniforge](https://github.com/conda-forge/miniforge)
1. create env:
    ```
    conda env create -f env.yml
    ```
1. activate env:
    ```
    conda activate rrg
    ```
1. install precommit:
    ```
    pre-commit install
    ```

## Table of Contents
* [Data Ingest](data-ingest.md): Data download and preprocessing
* [Experiments](experiments.md): Replicate experiments
* [CheXpert Labels](chexpert-labels.md): Classify reports
* [Inference](inference.md): Replicate inference results
