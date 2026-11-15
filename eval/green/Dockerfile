# GREEN (Ostmeier et al. 2024), unpatched, in the env its setup.py pins
# (requirements.txt says which commit). torch 2.2.2's wheel (CUDA 12.1) runs on H200.
# The judge comes from /models; its tokenizer is the model repo's own code
# (trust_remote_code, as upstream). Build and run it through eval/compose.yaml.

# python:3.12.1-slim-bookworm (GREEN's python_requires is ==3.12.1; every package
# installs from a wheel, nothing compiles).
FROM python:3.12.1-slim-bookworm@sha256:a64ac5be6928c6a94f00b16e09cdf3ba3edd44452d10ffa4516a58004873573e

COPY requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt
COPY --from=upstream . /opt/GREEN
# the submodule's .git points outside the image; upstream commits __pycache__
RUN rm -rf /opt/GREEN/.git \
    && find /opt/GREEN -name __pycache__ -prune -exec rm -rf {} + \
    && pip install --no-cache-dir --no-deps --no-build-isolation /opt/GREEN \
    && pip check \
    && pip freeze > /opt/pip-freeze.txt
# Last, so editing the wrapper doesn't redo the layers above.
COPY run.py /opt/labrag/run.py
ENTRYPOINT ["python", "/opt/labrag/run.py"]
