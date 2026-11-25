# CRIMSON (Baharoon et al. 2026), unpatched, in the env its pyproject.toml names
# (requirements.txt says how it is pinned). torch 2.10.0's wheel (CUDA 12.8) runs on H200.
# The judge comes from /models. Build and run it through eval/compose.yaml.

# python:3.12.16-slim-bookworm, the latest 3.12 patch release when pinned (2026-11-13);
# the multi-arch index digest. Every package installs from a wheel, nothing compiles.
FROM python:3.12.16-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

COPY requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt
COPY --from=upstream . /opt/CRIMSON
# the submodule's .git points outside the image; dependencies come from the lock above
RUN rm -rf /opt/CRIMSON/.git \
    && pip install --no-cache-dir --no-deps --no-build-isolation /opt/CRIMSON \
    && pip check \
    && pip freeze > /opt/pip-freeze.txt
# Last, so editing the wrapper doesn't redo the layers above.
COPY run.py /opt/labrag/run.py
ENTRYPOINT ["python", "/opt/labrag/run.py"]
