# RaTEScore (Zhao et al. 2024), unpatched, in ReXrank's env for it (requirements.txt);
# the submodule commit == PyPI 0.6.0, what ReXrank's `pip install ratescore` gets.
# torch 2.1.0's wheel (CUDA 12.1) runs on H200. Models come from /models.
# A report naming one entity with two types gives an exact tie that torch.topk breaks
# by device/threads, so such studies can differ across machines.
# Build and run it through eval/compose.yaml.

# python:3.10.22-slim-bookworm (every package installs from a wheel; nothing compiles).
FROM python:3.10.22-slim-bookworm@sha256:5be5aaecb962a41567bc8041f1e9e3432cb085b037fe057f9a562630b2e4b370

COPY requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt
COPY --from=upstream . /opt/RaTEScore
# the submodule's .git points outside the image
RUN rm -rf /opt/RaTEScore/.git \
    && pip install --no-cache-dir --no-deps --no-build-isolation /opt/RaTEScore \
    && pip check \
    && pip freeze > /opt/pip-freeze.txt
# Last, so editing the wrapper doesn't redo the layers above.
COPY run.py /opt/labrag/run.py
ENTRYPOINT ["python", "/opt/labrag/run.py"]
