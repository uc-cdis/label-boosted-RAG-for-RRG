# RaTEScore (Zhao et al. 2024), the implementation ReXrank uses, unpatched, in ReXrank's
# env for it (requirements.txt explains the pins). On one GPU, as ReXrank runs it: torch
# 2.1.0's PyPI wheel is the CUDA 12.1 build, which has kernels for our GPUs (H200, sm_90).
#
# Built from this directory plus the upstream submodule (external/RaTEScore, at the commit
# this repo pins, == PyPI RaTEScore 0.6.0, what ReXrank's `pip install ratescore` gets),
# both declared in eval/compose.yaml: build and run it from there.
#
# Model files come from /models, mounted read-only (the host weights root): run.py passes
# their paths to RaTEScore's constructor. None are baked into the image.
#
# Per-study scores can depend on the device and thread count: when a report names the
# same entity twice with different types (e.g. "pneumothorax" as DISEASE and as
# NON-DISEASE), RaTEScore's top-1 match is an exact tie that torch.topk breaks either way
# (checked 2026-11-13: 8 vs 16 CPU threads changed 1 of 200 pilot studies by 0.013).

# python:3.10.22-slim-bookworm (every package installs from a wheel; nothing compiles).
FROM python:3.10.22-slim-bookworm@sha256:5be5aaecb962a41567bc8041f1e9e3432cb085b037fe057f9a562630b2e4b370

COPY requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt
COPY --from=upstream . /opt/RaTEScore
# The submodule's .git is a pointer into this repo's .git, broken inside the image. The
# package is pure Python with no declared dependencies; built with the setuptools above.
RUN rm -rf /opt/RaTEScore/.git \
    && pip install --no-cache-dir --no-deps --no-build-isolation /opt/RaTEScore \
    && pip check \
    && pip freeze > /opt/pip-freeze.txt
# Last, so editing the wrapper doesn't redo the layers above.
COPY run.py /opt/labrag/run.py
ENTRYPOINT ["python", "/opt/labrag/run.py"]
