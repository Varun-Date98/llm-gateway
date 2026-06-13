from pathlib import Path

import httpx
import pytest

from gateway.backends.registry import ReplicaPool
from gateway.backends.vllm import VllmBackend, parse_chat_chunk, parse_sse_line
from gateway.config import GatewayConfig
from gateway.schemas import ChatCompletionRequest


def make_request(max_tokens: int = 2):
    return ChatCompletionRequest.model_validate(
        {"messages": [{"role": "user", "content": "hello"}], "max_tokens": max_tokens}
    ).to_generation_request(request_id="req_vllm")


def test_parse_sse_line_and_chat_chunk() -> None:
    payload = (
        'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}],'
        '"model":"mock"}'
    )

    event = parse_sse_line(payload)

    assert event is not None
    assert parse_chat_chunk(event) == ("hi", None)
    assert parse_sse_line("data: [DONE]") == "[DONE]"
    assert parse_sse_line(": keepalive") is None


@pytest.mark.asyncio
async def test_vllm_backend_streams_openai_sse_chunks() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = (
            'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
            'data: {"choices":[{"delta":{"content":"!"},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    backend = VllmBackend(
        model_id="llama-test",
        replica_id="llama-test-0",
        tier="small",
        endpoint="http://vllm.test",
        max_concurrency=2,
        transport=httpx.MockTransport(handler),
    )

    tokens = [token async for token in backend.generate(make_request())]

    assert [token.text for token in tokens] == ["hello", "!"]
    assert tokens[-1].finish_reason == "stop"
    assert backend.in_flight == 0


@pytest.mark.asyncio
async def test_vllm_backend_health_falls_back_to_models_endpoint() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(404)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(500)

    backend = VllmBackend(
        model_id="llama-test",
        replica_id="llama-test-0",
        tier="small",
        endpoint="http://vllm.test",
        max_concurrency=2,
        transport=httpx.MockTransport(handler),
    )

    assert await backend.health() is True
    assert backend.has_capacity is True


def test_registry_builds_vllm_replicas_from_config() -> None:
    config = GatewayConfig.model_validate(
        {
            "models": {
                "small": {
                    "id": "llama-test",
                    "tier": "small",
                    "backend": "vllm",
                    "endpoint": "http://127.0.0.1:8001",
                    "replicas": 2,
                }
            }
        }
    )

    pool = ReplicaPool.from_config(config)

    assert [replica.replica_id for replica in pool.replicas] == [
        "llama-test-0",
        "llama-test-1",
    ]
    assert all(isinstance(replica, VllmBackend) for replica in pool.replicas)


def test_vllm_example_config_loads() -> None:
    import yaml

    config = GatewayConfig.model_validate(
        yaml.safe_load(Path("config/gateway.vllm.example.yaml").read_text(encoding="utf-8"))
    )

    assert config.models["small"].backend == "vllm"
    assert config.models["large"].endpoint is not None
