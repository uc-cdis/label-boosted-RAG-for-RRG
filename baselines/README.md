# Literature baselines

Published report generators, each run as its authors run it, on our test targets: a
yardstick for the size of labrag − retrieval-only, not generators of ours. Each model
runs its upstream's own code, for data prep as well as inference, in its own container
with that upstream's pins, as the metrics in [eval/](../eval/README.md) do. Each model
gets the inputs its upstream pipeline gives it, not our prompts.

## Models
- `cxrmate-2`: CXRMate-2 (Nicolson et al. 2026), from
  [external/cxrmate-2](../external/cxrmate-2) and the `aehrc/cxrmate-2` checkpoint.
  Inputs: the study's images and the latest earlier study's (5 at most, sampled if
  more), the indication/history/comparison/technique of its report, and the earlier
  study's Findings/Impression. Greedy,
  fp32, batch size 1, as in upstream's test loop. [run.py](cxrmate-2/run.py)'s header
  lists the two additions that make outputs independent of the run.

## How they run
`labrag baseline --model <m> --dataset <mimic|rexgradient> --set test` builds the
image from the current checkout, checks the weights (sha256, `configs/weights.yaml`),
and runs two steps. `prepare` runs upstream's prepare script on CPU, into node-local
disk. `generate` runs one container per GPU of the job, each on its share of the
targets. Reports go to
`generations/test/<dataset>/findings+impression/baseline/<inputs>/<m>/test/`, at
generate's depth (`<inputs>` in the prompt's place: `upstream` = what the model's own
pipeline feeds it), so `labrag evaluate` and `compare` take them unchanged
([experiments/baselines-test.sh](../experiments/baselines-test.sh)).

Containers need Docker, so they run in Slurm jobs, never in the dev container.

## Rules
As in eval/: upstream is never forked. Only a small patch is applied at build time:
[cxrmate-2/labrag.patch](cxrmate-2/labrag.patch) reads our local ReXGradient copy
instead of downloading it, sorts each study's image paths, and fixes the MIMIC
script's save path (a NameError upstream). Pins come from the upstream, or are
resolved as of the day after the pinned commit. There is no network at run time:
weights come from `/models`, and the datasets are mounted read-only at the paths each
upstream expects (see [compose.yaml](compose.yaml)).

## Parity
CXRMate-2 published its own MIMIC-CXR and ReXGradient test reports. The baseline's
manifest counts how many targets have exactly the published Findings and Impression.
Studies with more than 5 images (current + prior) are counted on their own: the
processor samples 5 of their images at random, and upstream's random state is unknown.
