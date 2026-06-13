from fastapi.testclient import TestClient

from gateway.server import create_app


def test_health_endpoint_reports_running_scheduler() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["scheduler_running"] is True
    assert response.json()["replicas"] == 3


def test_models_endpoint_returns_configured_models() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    data = response.json()["data"]
    assert [model["id"] for model in data] == ["mock-small", "mock-large"]


def test_chat_completions_non_streaming() -> None:
    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "say hello"}],
                "max_tokens": 2,
            },
            headers={"x-request-id": "req_api_non_stream"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "req_api_non_stream"
    assert body["object"] == "chat.completion"
    assert body["model"] == "mock-small"
    assert body["choices"][0]["message"]["content"] == "mock token-1"
    assert body["usage"]["completion_tokens"] == 2


def test_chat_completions_streaming_sends_sse_chunks() -> None:
    with TestClient(create_app()) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "stream hello"}],
                "max_tokens": 2,
                "stream": True,
            },
            headers={"x-request-id": "req_api_stream"},
        ) as response:
            lines = [line for line in response.iter_lines() if line]

    assert response.status_code == 200
    assert lines[0].startswith("data: ")
    assert '"object": "chat.completion.chunk"' in lines[0]
    assert lines[-1] == "data: [DONE]"


def test_metrics_endpoint_exposes_gateway_metrics_after_request() -> None:
    with TestClient(create_app()) as client:
        client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "emit metrics"}],
                "max_tokens": 1,
            },
        )
        response = client.get("/metrics")

    assert response.status_code == 200
    body = response.text
    assert "gateway_http_requests_total" in body
    assert "gateway_admission_total" in body
    assert "gateway_tokens_total" in body
    assert "gateway_ttft_seconds" in body
