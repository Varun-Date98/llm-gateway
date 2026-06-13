import asyncio

import pytest

from gateway.backends.mock import MockBackend, MockLatencyModel
from gateway.schemas import ChatCompletionRequest


def make_request(max_tokens: int = 3):
    return ChatCompletionRequest.model_validate(
        {"messages": [{"role": "user", "content": "say hello"}], "max_tokens": max_tokens}
    ).to_generation_request(request_id="req_mock")


@pytest.mark.asyncio
async def test_mock_backend_streams_requested_tokens() -> None:
    backend = MockBackend(
        model_id="mock-small",
        replica_id="mock-small-0",
        tier="small",
        max_concurrency=4,
        latency_model=MockLatencyModel(time_scale=0.0, jitter_ratio=0.0),
    )

    tokens = [token async for token in backend.generate(make_request(max_tokens=3))]

    assert [token.index for token in tokens] == [0, 1, 2]
    assert all(token.model_id == "mock-small" for token in tokens)
    assert all(token.replica_id == "mock-small-0" for token in tokens)
    assert tokens[-1].is_final is True
    assert tokens[-1].finish_reason == "length"
    assert backend.in_flight == 0


@pytest.mark.asyncio
async def test_mock_backend_tracks_in_flight_during_generation() -> None:
    backend = MockBackend(
        model_id="mock-small",
        replica_id="mock-small-0",
        tier="small",
        max_concurrency=4,
        latency_model=MockLatencyModel(
            base_ttft_ms=20.0,
            base_itl_ms=0.0,
            prompt_ms_per_token=0.0,
            time_scale=1.0,
            jitter_ratio=0.0,
        ),
    )

    async def consume() -> None:
        try:
            async for _ in backend.generate(make_request(max_tokens=1)):
                pass
        finally:
            backend.release()

    assert backend.reserve() is True
    task = asyncio.create_task(consume())
    await asyncio.sleep(0)

    assert backend.in_flight == 1

    await task
    assert backend.in_flight == 0
