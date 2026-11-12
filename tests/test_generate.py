"""Generate: request building and record parsing against a fake vLLM server."""

import asyncio
import json

import httpx
import pytest

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
