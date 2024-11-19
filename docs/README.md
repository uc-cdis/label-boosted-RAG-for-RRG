# Label Boosted RAG for RRG
LaB-RAG (**La**bel **B**oosted **R**etrieval **A**ugmented **G**eneration): a generalizable framework for captioning using categorical labels as image descriptors. This repo contains code to reproduce our experiments on LaB-RAG in the context of radiology report generation (RRG).

### Environment setup
Using your favorite `conda` distribution (we prefer [miniforge](https://github.com/conda-forge/miniforge)), create and activate the environment:
```
conda env create -f env.yml
conda activate labrag
```

## Table of Contents
1. [Data Ingest](data-ingest.md): Data download and extraction
1. [Experiments](experiments.md): Replicate experiment results
1. [Baselines](baselines.md): Replicate baseline results
