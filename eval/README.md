# Evaluation metrics

Each metric runs its official upstream code, in its own container, with that upstream's
own pins. Nothing here reimplements a metric. Which metrics are used, and how our input
format differs from ReXrank's, is in [docs/revision-plan.md](../docs/revision-plan.md) §7.

## Services
All are defined in [compose.yaml](compose.yaml). Upstream code comes from the submodules
in `external/`.

- `cxr-report-metric`: BLEU-2, BERTScore, SembScore, RadGraph F1 and RadCliQ-v1
  (Yu et al. 2023). It uses the original pins (Python 3.7, torch 1.6) and runs on CPU only,
  because torch 1.6 computes wrong values on H200.
- `ratescore`: RaTEScore (Zhao et al. 2024), on one GPU, with ReXrank's env for it.
- `green-client` + `green-vllm`: GREEN (Ostmeier et al. 2024), as used for runs. The judge
  model is served by our pinned vLLM (`green-vllm`), and GREEN's own code does
  everything else: prompts, tokenization, parsing and scores (`green-client`).
- `green-native`: GREEN with upstream's own HF generate, on one GPU. This is the reference
  for the served path and is not used for runs.
- `crimson-client` + `crimson-vllm`, `crimson-native`: CRIMSON (Baharoon et al. 2026), the
  same pair of paths as GREEN's. CRIMSON's code builds the prompt and parses the judge's
  JSON; a study it can't parse or score gets `crimson_failed` = 1 and no score.

Each metric's directory holds its `Dockerfile`, its pins (`requirements.txt`, or
`constraints.txt` for CXR-Report-Metric) and `run.py`, the wrapper that calls upstream.
The header of each file says why its pins and settings are what they are.

## How they run
`labrag evaluate` builds the images from the current checkout and checks the model
weights (sha256, `configs/weights.yaml`) before any build. It formats the texts, runs the
services and writes the scores artifacts. Normally that is the only way they run.

Every `run.py` has the same interface:

```
<service> <gt.csv> <pred.csv> <out.csv>
```

- `gt.csv` and `pred.csv` have columns `study_id` (integers) and `report`, with the same
  ids in both and no empty reports. They are scored as given: header removal, whitespace
  collapse and the "EMPTY" placeholder are done by `labrag evaluate`, not here.
- `out.csv` has one row per study with the metric's scores. `<out>.json` holds the timing.

`green-client` and `crimson-client` also take `score <dir>...`, which judges each
directory's set (`gt.csv`, `pred.csv` → `out.csv`) alone on the running judge;
`labrag evaluate` uses it. The suite job runs the judges one at a time on its GPU.

To run one by hand (for debugging):

```
docker compose -f eval/compose.yaml build <service>
docker compose -f eval/compose.yaml run --rm -v <dir>:<dir> --user $(id -u):$(id -g) \
  -e OMP_NUM_THREADS=<n> <service> <dir>/gt.csv <dir>/pred.csv <dir>/out.csv
```

GPU services also need `CUDA_VISIBLE_DEVICES` set to one host GPU index. The served GREEN
path needs the judge running too (`green-client` waits for `green-vllm` to be healthy).

## Rules
- **Upstream is unpatched where possible.** Only CXR-Report-Metric has a patch,
  [labrag.patch](cxr-report-metric/labrag.patch), applied at build time: RadGraph runs on
  CPU (upstream hard-codes GPU 0), and every model loads from `/models`, because the
  weight CDN of transformers 3.0.1 is gone. Upstream is never forked.
- **Pins** are each upstream's own, or ReXrank's where the upstream has none. Packages
  the upstream leaves unpinned are resolved as of the day after the pinned commit, so a
  build today gets the tree the authors had.
- **No network at run time.** Running containers have no outside network (building an
  image does: pip installs the pins). Models come read-only from `/models`, which is
  `LABRAG_MODELS_ROOT` on the host. Only each judge and its client share a network, and
  it is internal.
- **Images by digest.** Base images are pinned by digest. The judge's vLLM image is never
  pulled for you (`pull_policy: never`): pull it by digest on each node in advance.
- **Images are tagged with the commit** (`LABRAG_COMMIT`, or `dev` for a dirty tree), and
  each job is its own compose project, so jobs at different commits don't collide.

## Reproducibility
- **CXR-Report-Metric:** scores change in the last float32 bit with the number of torch
  threads and with which studies share a run. Both settings are part of the scores'
  config in `labrag evaluate`, so they are recorded and fixed.
- **RaTEScore:** a report that names one entity with two types gives an exact tie, which
  `torch.topk` breaks differently by device and threads. Such studies can differ across
  machines.
- **GREEN, served:** `VLLM_BATCH_INVARIANT=1` and `TORCHINDUCTOR_DETERMINISTIC=1` make the
  judge's text independent of which studies share a batch and of the job.
  The judge's image, environment and command are part of the scores' config.
- **CRIMSON, served:** the same settings; whether they give the same guarantee for its
  judge (Gemma 3) is not measured yet.

GREEN's upstream reproduction test checks both GREEN paths. It is an opt-in GPU test; see
"Tests" in the [top-level README](../README.md).

## Adding a metric
1. Add the upstream as a submodule under `external/`, pinned to a commit.
2. Add `eval/<metric>/`: a `Dockerfile` that copies the upstream from the `upstream`
   build context, its pins, and a `run.py` with the interface above.
3. Add the service to `compose.yaml`: offline (`<<: *offline`), and `device_ids` from
   `CUDA_VISIBLE_DEVICES` if it needs a GPU.
4. Add its weights to `configs/weights.yaml` and its entry to `METRICS` in
   `src/labrag/stages/evaluate.py`.
