from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import httpx

from bench.scenarios import Scenario

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


@dataclass(frozen=True)
class RequestResult:
    scenario: str
    request_id: str
    status_code: int | None
    latency_seconds: float
    ttft_seconds: float | None
    model: str | None
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScenarioResult:
    scenario: dict[str, Any]
    requests: list[RequestResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "requests": [request.to_dict() for request in self.requests],
        }


async def run_open_loop(
    client: httpx.AsyncClient,
    scenario: Scenario,
    *,
    endpoint: str = CHAT_COMPLETIONS_PATH,
) -> ScenarioResult:
    """Run an open-loop load test at a fixed offered arrival rate."""

    total = scenario.request_count()
    interval = 1.0 / scenario.arrival_rate_rps
    semaphore = asyncio.Semaphore(scenario.max_concurrency or total)
    started_at = time.perf_counter()
    tasks: list[asyncio.Task[RequestResult]] = []

    for index in range(total):
        scheduled_at = started_at + index * interval
        await asyncio.sleep(max(0.0, scheduled_at - time.perf_counter()))
        tasks.append(
            asyncio.create_task(
                _bounded_request(
                    semaphore,
                    client,
                    scenario,
                    index,
                    endpoint=endpoint,
                )
            )
        )

    requests = await asyncio.gather(*tasks)
    return ScenarioResult(scenario=asdict(scenario), requests=list(requests))


async def _bounded_request(
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
    scenario: Scenario,
    index: int,
    *,
    endpoint: str,
) -> RequestResult:
    async with semaphore:
        return await issue_request(client, scenario, index, endpoint=endpoint)


async def issue_request(
    client: httpx.AsyncClient,
    scenario: Scenario,
    index: int,
    *,
    endpoint: str = CHAT_COMPLETIONS_PATH,
) -> RequestResult:
    request_id = f"{scenario.name}-{index}"
    payload = scenario.payload(index)
    started = time.perf_counter()
    try:
        if scenario.stream:
            return await _issue_streaming_request(
                client,
                endpoint,
                payload,
                scenario.name,
                request_id,
                started,
            )
        response = await client.post(endpoint, json=payload, headers={"x-request-id": request_id})
        latency = time.perf_counter() - started
        if response.status_code >= 400:
            return RequestResult(
                scenario=scenario.name,
                request_id=request_id,
                status_code=response.status_code,
                latency_seconds=latency,
                ttft_seconds=None,
                model=None,
                prompt_tokens=0,
                completion_tokens=0,
                error=response.text,
            )
        body = response.json()
        usage = body.get("usage", {})
        return RequestResult(
            scenario=scenario.name,
            request_id=request_id,
            status_code=response.status_code,
            latency_seconds=latency,
            ttft_seconds=latency,
            model=body.get("model"),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )
    except Exception as error:  # noqa: BLE001 - benchmark results should preserve failures.
        return RequestResult(
            scenario=scenario.name,
            request_id=request_id,
            status_code=None,
            latency_seconds=time.perf_counter() - started,
            ttft_seconds=None,
            model=None,
            prompt_tokens=0,
            completion_tokens=0,
            error=repr(error),
        )


async def _issue_streaming_request(
    client: httpx.AsyncClient,
    endpoint: str,
    payload: dict[str, Any],
    scenario_name: str,
    request_id: str,
    started: float,
) -> RequestResult:
    ttft: float | None = None
    model: str | None = None
    completion_tokens = 0
    async with client.stream(
        "POST",
        endpoint,
        json=payload,
        headers={"x-request-id": request_id},
    ) as response:
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                break
            if ttft is None:
                ttft = time.perf_counter() - started
            chunk = json.loads(data)
            model = chunk.get("model") or model
            completion_tokens += _content_token_count(chunk)

        latency = time.perf_counter() - started
        if response.status_code >= 400:
            return RequestResult(
                scenario=scenario_name,
                request_id=request_id,
                status_code=response.status_code,
                latency_seconds=latency,
                ttft_seconds=ttft,
                model=model,
                prompt_tokens=0,
                completion_tokens=completion_tokens,
                error=response.reason_phrase,
            )
        return RequestResult(
            scenario=scenario_name,
            request_id=request_id,
            status_code=response.status_code,
            latency_seconds=latency,
            ttft_seconds=ttft,
            model=model,
            prompt_tokens=estimate_prompt_tokens(payload.get("messages", [])),
            completion_tokens=completion_tokens,
        )


def estimate_prompt_tokens(messages: Iterable[dict[str, Any]]) -> int:
    text = "\n".join(f"{message.get('role')}: {message.get('content')}" for message in messages)
    return max(1, len(text.strip()) // 4) if text.strip() else 0


def _content_token_count(chunk: dict[str, Any]) -> int:
    choices = chunk.get("choices") or []
    if not choices:
        return 0
    content = (choices[0].get("delta") or {}).get("content") or ""
    return 1 if content else 0
