# GREEN (Ostmeier et al. 2024), the implementation ReXrank uses, unpatched, in the env
# its setup.py pins (requirements.txt explains which commit and why). On one GPU, as
# ReXrank runs it: torch 2.2.2's PyPI wheel is the CUDA 12.1 build, which has kernels for
# our GPUs (H200, sm_90).
#
# Built from this directory plus the upstream submodule (external/GREEN), both declared
# in eval/compose.yaml: build and run it from there.
#
# The judge (GREEN-RadLlama2-7b) comes from /models, mounted read-only (the host weights
# root): run.py passes its path to GREEN's constructor. Nothing is baked into the image.
# Its tokenizer is the model repo's own code (tokenization_chexagent.py), which
# transformers copies into HF_HOME/modules and imports (trust_remote_code, as upstream).
#
# Per-study scores can depend on batch neighbours: GREEN generates 8 reports at a time,
# left-padded, in fp16, so a study's text can change with the studies batched with it.
# The stage keeps the input order fixed (generation order).

# python:3.12.1-slim-bookworm (GREEN's python_requires is ==3.12.1; every package
# installs from a wheel, nothing compiles).
FROM python:3.12.1-slim-bookworm@sha256:a64ac5be6928c6a94f00b16e09cdf3ba3edd44452d10ffa4516a58004873573e

COPY requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt
COPY --from=upstream . /opt/GREEN
# The submodule's .git is a pointer into this repo's .git, broken inside the image;
# upstream also commits __pycache__ files. The package's dependencies are all in the
# lock above (pip check confirms its pins hold); built with the setuptools above.
RUN rm -rf /opt/GREEN/.git \
    && find /opt/GREEN -name __pycache__ -prune -exec rm -rf {} + \
    && pip install --no-cache-dir --no-deps --no-build-isolation /opt/GREEN \
    && pip check \
    && pip freeze > /opt/pip-freeze.txt
# Last, so editing the wrapper doesn't redo the layers above.
COPY run.py /opt/labrag/run.py
ENTRYPOINT ["python", "/opt/labrag/run.py"]
