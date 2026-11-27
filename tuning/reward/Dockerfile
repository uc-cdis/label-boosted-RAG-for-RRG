# CXRMate-2's GRPO rewards (Nicolson et al. 2026): upstream's rewards.py, unpatched,
# with RaTEScore, in the env of requirements.txt, served by server.py. torch 2.11.0's
# wheel (CUDA 13.0) runs on the nodes' 580 driver, as baselines/cxrmate-2. Build and run
# it through tuning/compose.yaml.

# python:3.12.16-slim-bookworm, as baselines/cxrmate-2 (every package installs from a
# wheel).
FROM python:3.12.16-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

COPY requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt \
    && pip check \
    && pip freeze > /opt/pip-freeze.txt
COPY --from=upstream rewards.py /opt/cxrmate-2/rewards.py
# Last, so editing the server doesn't redo the layers above.
COPY server.py /opt/labrag/server.py
ENV PYTHONPATH=/opt/cxrmate-2
ENTRYPOINT ["python", "/opt/labrag/server.py"]
