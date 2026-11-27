# GRPO trainer: TRL's GRPOTrainer (trl.patch: CXRMate-2's unscaled advantage) with PEFT
# LoRA and vLLM colocated for the rollouts, on the pinned vLLM image (configs/
# containers.yaml), so rollouts run the vLLM generate serves with. requirements.txt
# adds to its env without changing any package it has. reports.py: labrag's, so the
# reward scores the text evaluate scores. Build and run it through tuning/compose.yaml.

FROM vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34

RUN apt-get update && apt-get install -y --no-install-recommends patch \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /opt/labrag/requirements.txt
# The image's own pins are the constraints; new packages as of the day pinned.
RUN python3 -m pip install --no-cache-dir uv==0.13.7 \
    && python3 -m pip freeze | grep '==' > /opt/labrag/vllm-image-freeze.txt \
    && uv pip install --system --break-system-packages --no-cache \
        --exclude-newer 2026-11-28 -c /opt/labrag/vllm-image-freeze.txt \
        -r /opt/labrag/requirements.txt \
    && python3 -m pip check \
    && python3 -m pip freeze > /opt/pip-freeze.txt
COPY trl.patch /opt/labrag/trl.patch
RUN patch -p1 --forward -d "$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
    < /opt/labrag/trl.patch
COPY --from=labrag reports.py /opt/labrag/reports.py
# Last, so editing the wrapper doesn't redo the layers above.
COPY run.py /opt/labrag/run.py
ENV PYTHONPATH=/opt/labrag
ENTRYPOINT ["python3", "/opt/labrag/run.py"]
