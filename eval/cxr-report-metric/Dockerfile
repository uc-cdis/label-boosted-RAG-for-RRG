# CXR-Report-Metric (Yu et al. 2023): BLEU-2, BERTScore, SembScore, RadGraph F1,
# RadCliQ-v1, with the original pins (Python 3.7, torch 1.6.0, transformers 3.0.1,
# allennlp 1.1.0), not ReXrank-metric's env, where RadGraph's checkpoint doesn't load.
# CPU only: torch 1.6 has no sm_90 kernels (on H200 ops fail, index_select silently
# returns wrong values). Never pass --gpus.
#
# labrag.patch runs RadGraph on CPU (upstream hard-codes GPU 0) and points every model
# to /models (transformers 3.0.1's weight CDN is gone; jobs have no network).
# Build and run it through eval/compose.yaml (it adds the upstream submodule).

# python:3.7.17-bookworm: last 3.7, has gcc for the sdists. Its pip 23.0.1 stays:
# pip >= 24.1 rejects bert-score 0.3.11's malformed metadata.
FROM python:3.7.17-bookworm@sha256:eedf63967cdb57d8214db38ce21f105003ed4e4d0358f02bedc057341bcf92a0

WORKDIR /opt/CXR-Report-Metric
COPY --from=upstream requirements.txt .
# constraints.txt pins the rest of the tree as of the upstream commit
COPY constraints.txt /opt/constraints.txt
RUN pip install --no-cache-dir -r requirements.txt -c /opt/constraints.txt \
    && pip freeze > /opt/pip-freeze.txt
COPY --from=upstream . .
COPY labrag.patch /opt/labrag.patch
# without the submodule's .git, git apply patches plain files (and fails on a bad hunk)
RUN rm -rf .git && git apply --verbose /opt/labrag.patch
# upstream writes caches into its working directory; jobs run as their own uid
RUN chmod -R a+rwX /opt/CXR-Report-Metric
ENV MKL_THREADING_LAYER=GNU
# Last, so editing the wrapper doesn't redo the layers above.
COPY run.py /opt/labrag/run.py
ENTRYPOINT ["python", "/opt/labrag/run.py"]
