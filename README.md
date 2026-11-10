# Label Boosted RAG for RRG

LaB-RAG (**La**bel **B**oosted **R**etrieval **A**ugmented **G**eneration): small models
predict structured labels from the image, and the labels boost frozen, text-only LLMs
for radiology report generation (RRG).

This branch is a rewrite for the journal revision; see
[docs/revision-plan.md](docs/revision-plan.md). The code for the original preprint is
on `main` (tag `paper-v1`).

## Layout
- `src/labrag/`: the pipeline package (`labrag` CLI)
- `uv.lock`, `.python-version`: the env, managed by [uv](https://docs.astral.sh/uv/).
  `envs/install-uv.sh` installs the pinned uv, then `uv sync --frozen` builds `.venv/`
  exactly from the lock (`--no-dev` for runs). vLLM and the metrics run as containers.
- `slurm/`: `submit.sh [sbatch options] -- <command>` runs a command on the cluster at the
  current commit (clean tree, pushed). Each node keeps one shallow clone plus env per
  commit under `~/labrag-run/commits/<sha>/`, so jobs at different commits never interfere.
- `external/`: upstream code as git submodules (`mimic-cxr`: MIT-LCP's report section parser)
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
