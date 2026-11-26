# CXRMate-2 (Nicolson et al. 2026): upstream's prepare scripts and the checkpoint's own
# modelling/processing code (trust_remote_code, from /models; byte-identical to the
# upstream commit's) in the env requirements.txt pins. labrag.patch, applied at build
# time, makes the ReXGradient script read our local copy instead of downloading it and
# sort each study's images, and fixes the MIMIC script's save path (a NameError).
# torch 2.11.0's wheel (CUDA 13.0) runs on the nodes' 580 driver. Build and run it
# through baselines/compose.yaml.

# The patch, applied where git is: eval/cxr-report-metric's pinned base. Without the
# submodule's .git, git apply patches plain files (and fails on a bad hunk).
FROM python:3.7.17-bookworm@sha256:eedf63967cdb57d8214db38ce21f105003ed4e4d0358f02bedc057341bcf92a0 AS patched
COPY --from=upstream . /src
COPY labrag.patch /labrag.patch
RUN rm -rf /src/.git && cd /src && git apply --verbose /labrag.patch

# python:3.12.16-slim-bookworm, as eval/crimson (every package installs from a wheel).
FROM python:3.12.16-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

COPY requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt \
    && pip check \
    && pip freeze > /opt/pip-freeze.txt
COPY --from=patched /src /opt/cxrmate-2
# Last, so editing the wrapper doesn't redo the layers above.
COPY run.py /opt/labrag/run.py
ENTRYPOINT ["python", "/opt/labrag/run.py"]
