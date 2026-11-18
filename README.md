# Label Boosted RAG for RRG

LaB-RAG (**La**bel **B**oosted **R**etrieval **A**ugmented **G**eneration): small models
predict structured labels from the image, and the labels boost frozen, text-only LLMs
for radiology report generation (RRG).

This branch is a rewrite for the journal revision; see
[docs/revision-plan.md](docs/revision-plan.md). The code for the original preprint is
on `main` (tag `paper-v1`).

## Pipeline
`labrag <command>`, in order (`labrag <command> --help` for options):

- `labels`: reference labels on each study's report section
- `embed`: one image embedding per study
- `classify`: per-label image classifiers
- `build-prompts`: one chat prompt per target study, for one condition
- `generate`: one generation per prompt, from a model served by vLLM
- `evaluate`: per-study scores of generated reports, by official metric code
- `compare`: means with 95% CIs per condition, and paired differences

Each stage writes an artifact: a directory with a `manifest.json` (commit, config,
inputs and their checksums). The same name with the same config is skipped, a different
config is refused, and `--overwrite` replaces it. An artifact whose inputs changed is
stale and refused. Two more commands: `labrag provenance` lists every stale artifact,
and `labrag splits --dataset <name>` redraws a dataset's frozen split files.

## Layout
- `src/labrag/`: the pipeline package (`labrag` CLI); one module per stage in
  `stages/`, one adapter per dataset in `data/`
- `configs/`: models, sampling, prompt versions, container images (by digest) and
  model weights (by sha256)
- `eval/`: one container per evaluation metric, each with its upstream's own pins
  ([eval/README.md](eval/README.md)); `evaluate` builds and runs them
- `external/`: upstream code as git submodules (MIT-LCP's `mimic-cxr` report section
  parser, `CheXbert`, and the metrics `CXR-Report-Metric`, `RaTEScore`, `GREEN`)
- `uv.lock`, `.python-version`: the env, managed by [uv](https://docs.astral.sh/uv/).
  `envs/install-uv.sh` installs the pinned uv, then `uv sync --locked` builds `.venv/`
  exactly from the lock. Jobs sync `--no-dev --group test` (the run deps plus pytest).
  vLLM and the metrics run as containers.
- `slurm/`: `submit.sh [sbatch options] -- <command>` runs a command on the cluster at the
  current commit (clean tree, pushed). Each node keeps one shallow clone plus env per
  commit under `~/labrag-run/commits/<sha>/`, so jobs at different commits never interfere.
  Every job also leaves `~/labrag-run/bin/hf` on its nodes (`envs/install-hf.sh`, versions
  from the lock) for pre-pulling weights by hand; jobs themselves never download.
- `splits/`: frozen, committed split files with provenance ([splits/README.md](splits/README.md))
- `resources/`: small hand-made tables (the PadChest location hierarchy)
- `tests/`: checks (see Tests)
- `docs/`: design docs

## Data and storage
Each root is set by an environment variable (`src/labrag/paths.py`):

- `LABRAG_CXR_ROOT` (default `/opt/gpudata/cxr`): dataset releases, read-only. Only the
  source files released by the dataset authors are read; everything derived from them
  is produced by this repo.
- `LABRAG_MODELS_ROOT` (default `/opt/gpudata/models`): model weights, pre-pulled on
  every node and checked against `configs/weights.yaml`.
- `LABRAG_SHARED_ROOT` (default `/shared/gpudata/songs1/labrag-shared`): shared storage
  for every artifact. Jobs build on local disk and publish here once; each job's log
  is copied to `logs/<job-id>.out` when it ends.
- `LABRAG_RUN_ROOT` (default `~/labrag-run`): node-local commit checkouts, caches and
  job logs.

## Tests
`uv run pytest` runs the checks, including exact reproduction of the frozen splits.
Tests that need a dataset skip when it isn't mounted.

Tests marked `gpu` run only when asked for with `-m gpu`. GREEN's upstream reproduction
test, through both of our judge paths, is one of them. On the cluster:

```
slurm/submit.sh --gres=gpu:1 -- pytest -p no:cacheprovider -m gpu tests/test_green_repro.py
```

On any other GPU machine, name one GPU by its host index:

```
CUDA_VISIBLE_DEVICES=0 uv run pytest -m gpu
```

The machine needs Docker with the compose plugin and the NVIDIA container toolkit, the
submodules checked out, the GREEN judge weights under `LABRAG_MODELS_ROOT` (checked
against `configs/weights.yaml`) and the vLLM image from `eval/compose.yaml` pulled by
digest; images are never pulled for you.

## Citation
```
TODO: add bibtex citation
```
