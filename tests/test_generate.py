"""Generate: request building and record parsing against a fake vLLM server."""

import asyncio
import base64
import io
import json
import subprocess

import httpx
import numpy as np
import pytest
from PIL import Image

from labrag import serving
from labrag.stages import generate


class FakeTokenizer:
    class _Enc:
        def __init__(self, n):
            self.ids = list(range(n))

    def encode(self, text, add_special_tokens=False):
        return self._Enc(len(text.split()))


def test_requests_and_records(monkeypatch):
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        i = len(seen)
        truncated = body["messages"][0]["content"] == "long"
        message = {
            "role": "assistant",
            "content": None if truncated else f"Report {i}.",
            "reasoning": "think think think",
        }
        return httpx.Response(
            200,
            json={
                "id": f"c{i}",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "length" if truncated else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 7,
                    "total_tokens": 17,
                },
            },
        )

    real = generate.AsyncOpenAI
    monkeypatch.setattr(
        generate,
        "AsyncOpenAI",
        lambda **kw: real(
            **kw, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        ),
    )
    sampling = serving.load_config("sampling")["qwen3.5-thinking-32k"]
    rows = [
        {
            "study_id": "a",
            "set": "pilot",
            "messages": [{"role": "user", "content": "x"}],
        },
        {
            "study_id": "b",
            "set": "pilot",
            "messages": [{"role": "user", "content": "long"}],
        },
    ]
    out = asyncio.run(
        generate._generate("http://test/v1", "m", sampling, rows, FakeTokenizer())
    )
    body = seen[0]
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["seed"] == sampling["seed"] and body["max_tokens"] == 32768
    for k, v in sampling["params"].items():
        assert body[k] == pytest.approx(v), k
    assert [r["study_id"] for r in out] == ["a", "b"]
    assert (
        out[0]["reasoning"] == "think think think" and out[0]["reasoning_tokens"] == 3
    )
    assert out[1]["report"] is None and out[1]["finish_reason"] == "length"
    assert all(r["latency_s"] >= 0 and r["start_s"] >= 0 for r in out)
    s = generate.summarize(out)
    assert s["finish_reason"] == {"length": 1, "stop": 1} and s["n_no_report"] == 1


def test_request_messages_swap_in_the_prepared_image(tmp_path):
    text = {"role": "user", "content": "x"}
    assert generate.request_messages({"messages": [text]}, tmp_path) == [text]
    (tmp_path / "d").mkdir()
    pixels = np.arange(400 * 300, dtype=np.uint16).reshape(300, 400)  # 16-bit, wide
    Image.fromarray(pixels).save(tmp_path / "d/p.png")
    parts = [{"type": "text", "text": "a"}, {"type": "image"}]
    row = {"image": "d/p.png", "messages": [{"role": "user", "content": parts}]}
    for short_edge, size in [(150, (200, 150)), (None, (400, 300))]:
        [m] = generate.request_messages(row, tmp_path, short_edge)
        assert m["content"][0] == {"type": "text", "text": "a"}
        url = m["content"][1]["image_url"]["url"]
        assert m["content"][1]["type"] == "image_url"
        prefix = "data:image/png;base64,"
        assert url.startswith(prefix)
        sent = Image.open(io.BytesIO(base64.b64decode(url[len(prefix) :])))
        assert sent.mode == "L" and sent.size == size
        assert np.asarray(sent).min() == 0 and np.asarray(sent).max() == 255
    assert row["messages"][0]["content"][1] == {"type": "image"}  # not mutated


def test_configs_consistent():
    models = serving.load_config("models")
    containers = serving.load_config("containers")
    from labrag import weights

    for name, m in models.items():
        assert m["container"] in containers, name
        assert m["weights"] in weights.load_config(), name
    for c in containers.values():
        assert c["digest"].startswith("sha256:") and len(c["digest"]) == 71


def test_counter_totals_sums_labels_and_keeps_only_vllm_totals():
    text = """# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{engine="0",model_name="m"} 1200.0
vllm:prompt_tokens_total{engine="1",model_name="m"} 300.0
vllm:generation_tokens_total{engine="0",model_name="m"} 5e3
vllm:e2e_request_latency_seconds_count{engine="0",model_name="m"} 4.0
python_gc_objects_collected_total{generation="0"} 9.0
"""
    assert serving.counter_totals(text) == {
        "vllm:generation_tokens_total": 5000.0,
        "vllm:prompt_tokens_total": 1500.0,
    }


@pytest.mark.parametrize(
    "returncode, stderr, expected",
    [
        (0, "", "sha256:abc"),
        (1, "Error response from daemon: No such image: x@sha256:1", None),
        (1, "permission denied while trying to connect to the docker API", SystemExit),
    ],
)
def test_image_id_only_missing_image_is_none(monkeypatch, returncode, stderr, expected):
    monkeypatch.setattr(serving.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(
        serving.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a, returncode, "sha256:abc\n", stderr
        ),
    )
    if expected is SystemExit:
        with pytest.raises(SystemExit, match="permission denied"):
            serving.image_id("x@sha256:1")
    else:
        assert serving.image_id("x@sha256:1") == expected
