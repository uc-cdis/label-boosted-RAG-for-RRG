# CRIMSON (Baharoon et al. 2026), the rajpurkarlab/CRIMSON commit this repo pins as a
# submodule, unpatched, in an env pinned to the versions its pyproject.toml names
# (requirements.txt explains how). torch 2.10.0's PyPI wheel is the CUDA 12.8 build, which
# has kernels for our GPUs (H200, sm_90).
#
# Built from this directory plus the upstream submodule (external/CRIMSON), both declared
# in eval/compose.yaml: build and run it from there.
#
# The judge (medgemma-4b-it-crimson) comes from /models, mounted read-only (the host
# weights root): run.py passes its path to CRIMSONScore. Nothing is baked into the image.

# python:3.12.16-slim-bookworm, the latest 3.12 patch release when pinned (2026-11-13);
# the multi-arch index digest. Every package installs from a wheel, nothing compiles.
FROM python:3.12.16-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

COPY requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt
COPY --from=upstream . /opt/CRIMSON
# The submodule's .git is a pointer into this repo's .git, broken inside the image. The
# package's dependencies are all in the lock above (pip check confirms they hold); built
# with the setuptools and wheel above.
RUN rm -rf /opt/CRIMSON/.git \
    && pip install --no-cache-dir --no-deps --no-build-isolation /opt/CRIMSON \
    && pip check \
    && pip freeze > /opt/pip-freeze.txt
# Last, so editing the wrapper doesn't redo the layers above.
COPY run.py /opt/labrag/run.py
ENTRYPOINT ["python", "/opt/labrag/run.py"]
