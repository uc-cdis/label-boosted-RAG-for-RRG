# CXR-Report-Metric (Yu et al. 2023; BLEU-2, BERTScore, SembScore, RadGraph F1, RadCliQ-v1),
# the official implementation ReXrank uses, with its own pins: Python 3.7, torch 1.6.0
# (the PyPI wheel is the CUDA 10.2 build), transformers 3.0.1, allennlp 1.1.0.
# Not ReXrank-metric's env (torch 1.13.1, transformers 4.24.0; checked 2026-11-13 on 200
# pilot studies): there allennlp can't load the RadGraph checkpoint, and bert-score 0.3.11
# no longer recognizes the RoBERTa tokenizer, so BERTScore tokenizes the first word
# differently. BLEU-2 and SembScore agree.
#
# Built from this directory plus the upstream submodule (external/CXR-Report-Metric, at the
# commit this repo pins), both declared in eval/compose.yaml: build and run it from there.
#
# CPU only: torch 1.6 has no kernels for our GPUs (H200, sm_90; a GPU smoke test on kg35-nvl01,
# 2026-11-13: ops fail, and index_select silently returned wrong values). Never pass --gpus.
# On CPU, RadGraph (allennlp, one report at a time) is the slow step: ~0.1 s/report at 16
# threads (run.py explains the thread count).
#
# labrag.patch runs RadGraph on CPU (upstream hard-codes GPU 0) and otherwise changes only
# where model files come from: every Hub name or relative path becomes a local path under
# /models, mounted read-only (the host weights root; eval/compose.yaml).
# transformers 3.0.1 can't download them any more (its weight CDN is gone) and jobs run with
# no network. No model files are baked into the image.

# python:3.7.17-bookworm (3.7.17 is the last 3.7 release; the image has gcc, which the
# sdists below need: jsonnet, fast-bleu). Its pip (23.0.1) is kept on purpose: pip >= 24.1
# rejects bert-score 0.3.11's malformed metadata ("transformers>=3.0.0numpy").
FROM python:3.7.17-bookworm@sha256:eedf63967cdb57d8214db38ce21f105003ed4e4d0358f02bedc057341bcf92a0

WORKDIR /opt/CXR-Report-Metric
COPY --from=upstream requirements.txt .
# constraints.txt pins the rest of the tree as of the upstream commit (see its header);
# the freeze records what was installed.
COPY constraints.txt /opt/constraints.txt
RUN pip install --no-cache-dir -r requirements.txt -c /opt/constraints.txt \
    && pip freeze > /opt/pip-freeze.txt
COPY --from=upstream . .
COPY labrag.patch /opt/labrag.patch
# The submodule's .git is a pointer into this repo's .git, broken inside the image; with
# it gone, git apply patches plain files (and fails the build if any hunk doesn't apply).
RUN rm -rf .git && git apply --verbose /opt/labrag.patch
# Upstream writes its caches and temp files into the working directory, and jobs run as
# the submitting user, not root.
RUN chmod -R a+rwX /opt/CXR-Report-Metric
ENV MKL_THREADING_LAYER=GNU
# Last, so editing the wrapper doesn't redo the layers above.
COPY run.py /opt/labrag/run.py
ENTRYPOINT ["python", "/opt/labrag/run.py"]
