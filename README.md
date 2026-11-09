# Label Boosted RAG for RRG

LaB-RAG (**La**bel **B**oosted **R**etrieval **A**ugmented **G**eneration): small models
predict structured labels from the image, and the labels boost frozen, text-only LLMs
for radiology report generation (RRG).

This branch is a rewrite for the journal revision; see
[docs/revision-plan.md](docs/revision-plan.md). The code for the original preprint is
on `main` (tag `paper-v1`).

## Layout
- `src/labrag/`: the pipeline package (`labrag` CLI)
- `splits/`: frozen, committed split files with provenance ([splits/README.md](splits/README.md))
- `tests/`: checks, including exact reproduction of the frozen splits
- `docs/`: design docs

## Data
Only source files released by the dataset authors are read (root: `LABRAG_CXR_ROOT`,
default `/opt/gpudata/cxr`). Everything derived from them is produced by this repo.
Outputs go to `LABRAG_OUT_ROOT` (default `/home/songs1/labrag-outputs`).

## Citation
```
TODO: add bibtex citation
```
