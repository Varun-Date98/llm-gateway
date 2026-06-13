from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from gateway.config import ModelConfig
from gateway.schemas import GenerationRequest, Token


class VllmBackend:
    """OpenAI-compatible streaming client for a vLLM replica."""

    def __init__(
        self,
        *,
        model_id: str,
        replica_id: str,
        tier: str,
        endpoint: str,
        max_concurrency: int,
        api_key: str | None = None,
        request_timeout_seconds: float = 120.0,
        health_path: str = "/health",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model_id = model_id
        self.replica_id = replica_id
        self.tier = tier
        self.endpoint = endpoint.rstrip("/")
        self.max_concurrency = max_concurrency
        self.api_key = api_key
        self.request_timeout_seconds = request_timeout_seconds
        self.health_path = health_path
        self.transport = transport
        self._in_flight = 0
        self._healthy = True
        self._client = self._build_client()

    @classmethod
    def from_model_config(
        cls,
        model: ModelConfig,
        *,
        replica_index: int,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> VllmBackend:
        if model.endpoint is None:
            raise ValueError("vLLM backend requires an endpoint")
        api_key = os.getenv(model.api_key_env) if model.api_key_env else None
        return cls(
            model_id=model.id,
            replica_id=f"{model.id}-{replica_index}",
            tier=model.tier,
            endpoint=str(model.endpoint),
            max_concurrency=model.max_concurrency,
            api_key=api_key,
            request_timeout_seconds=model.request_timeout_seconds,
            health_path=model.health_path,
            transport=transport,
        )

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def has_capacity(self) -> bool:
        return self._healthy and self._in_flight < self.max_concurrency

    async def health(self) -> bool:
        try:
            response = await self._client.get(self.health_path)
            if response.status_code == 404:
                response = await self._client.get("/v1/models")
            self._healthy = response.status_code == 200
        except httpx.HTTPError:
            self._healthy = False
        return self._healthy

    def reserve(self) -> bool:
        if not self.has_capacity:
            return False
        self._in_flight += 1
        return True

    def release(self) -> None:
        self._in_flight = max(0, self._in_flight - 1)

    async def close(self) -> None:
        await self._client.aclose()

    def generate(self, request: GenerationRequest) -> AsyncIterator[Token]:
        return self._generate(request)

    async def _generate(self, request: GenerationRequest) -> AsyncIterator[Token]:
        index = 0
        completion_tokens = 0
        async with self._client.stream(
            "POST",
            "/v1/chat/completions",
            json=self._payload(request),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                event = parse_sse_line(line)
                if event is None:
                    continue
                if event == "[DONE]":
                    break
                token_text, finish_reason = parse_chat_chunk(event)
                if not token_text and finish_reason is None:
                    continue
                completion_tokens += 1 if token_text else 0
                yield Token(
                    text=token_text,
                    index=index,
                    model_id=self.model_id,
                    replica_id=self.replica_id,
                    request_id=request.request_id,
                    is_final=finish_reason is not None,
                    finish_reason=finish_reason,
                    prompt_tokens=request.prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                index += 1

    def _payload(self, request: GenerationRequest) -> dict[str, Any]:
        payload = dict(request.raw_request)
        payload["model"] = request.requested_model or self.model_id
        payload["messages"] = [message.model_dump(mode="json") for message in request.messages]
        payload["max_tokens"] = request.max_tokens
        payload["temperature"] = request.temperature
        payload["top_p"] = request.top_p
        payload["stream"] = True
        if request.stop is not None:
            payload["stop"] = request.stop
        if request.user is not None:
            payload["user"] = request.user
        return payload

    def _build_client(self) -> httpx.AsyncClient:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        return httpx.AsyncClient(
            base_url=self.endpoint,
            headers=headers,
            timeout=self.request_timeout_seconds,
            transport=self.transport,
        )


def parse_sse_line(line: str) -> str | None:
    line = line.strip()
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None
    return line.removeprefix("data:").strip()


def parse_chat_chunk(payload: str) -> tuple[str, str | None]:
    data = json.loads(payload)
    choices = data.get("choices") or []
    if not choices:
        return "", None
    choice = choices[0]
    delta = choice.get("delta") or {}
    return delta.get("content") or "", choice.get("finish_reason")
