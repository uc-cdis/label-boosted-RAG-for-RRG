"""CXRMate-2's GRPO rewards over HTTP, for the trainer in the grpo image.

The rewards are upstream's (rewards.py and RaTEScore, unpatched), built and called as
stages_cxrmate2_grpo.py's post_prepare and advantages do (non-FSDP path): RaTEScore per
pair (a failure or None scores 0.0), CXR-BERT cosine, BERTScore and ARN (repeated
3-grams, with CXRMate-2's own tokenizer) per group of completions of one prompt. Only
the raw rewards: the trainer z-scores them per group and sums them with the weights.

Upstream loads its models by hub id. Offline, an HF cache under HF_HOME maps each id to
its full clone under /models (symlinks; the revision from the clone's download
metadata), so no upstream code changes.

POST /score {"predictions": [...], "references": [...], "group_size": n}: lists in
group order (n consecutive predictions per prompt) -> {"ratescore": [...], "cxrbert":
[...], "bertscore": [...], "arn": [...]}. GET /health: 200 once loaded. Serves one
request at a time on the container's first GPU.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PORT = int(sys.argv[1])
MODELS = Path("/models")
# hub id as upstream names it -> full clone under /models
HUB_IDS = {
    "roberta-large": "FacebookAI/roberta-large",
    "microsoft/BiomedVLP-CXR-BERT-specialized": "microsoft/BiomedVLP-CXR-BERT-specialized",
    "Angelakeke/RaTE-NER-Deberta": "Angelakeke/RaTE-NER-Deberta",
    "FremyCompany/BioLORD-2023-C": "FremyCompany/BioLORD-2023-C",
}
CXRMATE2 = MODELS / "aehrc/cxrmate-2"  # its tokenizer.json is upstream's processor's


def offline_cache(hub: Path) -> None:
    """HF_HOME/hub entries for HUB_IDS, each file a symlink into its clone."""
    for hub_id, local in HUB_IDS.items():
        clone = MODELS / local
        revision = clone / ".cache/huggingface/download/config.json.metadata"
        revision = revision.read_text().splitlines()[0]
        repo = hub / ("models--" + hub_id.replace("/", "--"))
        (repo / "refs").mkdir(parents=True, exist_ok=True)
        (repo / "refs/main").write_text(revision)
        snapshot = repo / "snapshots" / revision
        for f in clone.rglob("*"):
            rel = f.relative_to(clone)
            if f.is_file() and rel.parts[0] != ".cache":
                (snapshot / rel).parent.mkdir(parents=True, exist_ok=True)
                (snapshot / rel).symlink_to(f)


offline_cache(Path(os.environ["HF_HOME"]) / "hub")

import torch  # noqa: E402
from RaTEScore import RaTEScore  # noqa: E402
from rewards import ARNReward, BERTScoreReward, CXRBERTReward  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

device = torch.device("cuda")
ratescore = RaTEScore()
ratescore.model.to(device=device)
ratescore.eval_model.to(device=device)
cxrbert = CXRBERTReward(device=device)
bertscore = BERTScoreReward(
    device=device, num_workers=int(os.environ["OMP_NUM_THREADS"])
)
arn = ARNReward(n=3, device=device, tokenizer=AutoTokenizer.from_pretrained(CXRMATE2))


def rate(prediction: str, reference: str) -> float:
    try:
        score = ratescore.compute_score([prediction], [reference])
        return score[0] if score is not None else 0.0
    except Exception:  # upstream's rule: a failed score is 0
        return 0.0


def score(predictions: list[str], references: list[str], group_size: int) -> dict:
    out = {"ratescore": [], "cxrbert": [], "bertscore": [], "arn": []}
    for start in range(0, len(predictions), group_size):
        p = predictions[start : start + group_size]
        r = references[start : start + group_size]
        with torch.no_grad():
            out["ratescore"] += [float(rate(i, j)) for i, j in zip(p, r, strict=True)]
            out["cxrbert"] += cxrbert(p, r).tolist()
            out["bertscore"] += bertscore(p, r).tolist()
            out["arn"] += arn(p).tolist()
    return out


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._reply(200, {"ok": True} if self.path == "/health" else None)

    def do_POST(self):
        if self.path != "/score":
            return self._reply(404, None)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if len(body["predictions"]) != len(body["references"]):
            return self._reply(
                400, {"error": "predictions and references differ in length"}
            )
        self._reply(
            200, score(body["predictions"], body["references"], body["group_size"])
        )

    def _reply(self, code: int, payload) -> None:
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # one line per request is noise in the job log
        pass


print(f"rewards ready on port {PORT}", flush=True)
HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
