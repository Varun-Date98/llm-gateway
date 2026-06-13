from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass

from gateway.config import ModelConfig
from gateway.schemas import GenerationRequest, Token


@dataclass(frozen=True)
class MockLatencyModel:
    """Latency model that gets slower as prompts and replica concurrency grow."""

    base_ttft_ms: float = 25.0
    prompt_ms_per_token: float = 0.03
    base_itl_ms: float = 8.0
    concurrency_penalty: float = 0.35
    jitter_ratio: float = 0.05
    time_scale: float = 1.0

    def sample_ttft(self, prompt_tokens: int, in_flight: int) -> float:
        latency_ms = self.base_ttft_ms + (prompt_tokens * self.prompt_ms_per_token)
        return self._to_seconds(latency_ms, in_flight)

    def sample_itl(self, in_flight: int) -> float:
        return self._to_seconds(self.base_itl_ms, in_flight)

    def _to_seconds(self, latency_ms: float, in_flight: int) -> float:
        load_multiplier = 1.0 + max(0, in_flight - 1) * self.concurrency_penalty
        jitter = 1.0
        if self.jitter_ratio > 0:
            jitter += random.uniform(-self.jitter_ratio, self.jitter_ratio)
        return max(0.0, latency_ms * load_multiplier * jitter * self.time_scale / 1000.0)


class MockBackend:
    """CPU-free backend that streams tokens while mimicking load-sensitive latency."""

    def __init__(
        self,
        *,
        model_id: str,
        replica_id: str,
        tier: str,
        max_concurrency: int,
        latency_model: MockLatencyModel | None = None,
        healthy: bool = True,
    ) -> None:
        self.model_id = model_id
        self.replica_id = replica_id
        self.tier = tier
        self.max_concurrency = max_concurrency
        self.latency_model = latency_model or MockLatencyModel()
        self._healthy = healthy
        self._in_flight = 0

    @classmethod
    def from_model_config(
        cls,
        model: ModelConfig,
        *,
        replica_index: int,
        latency_model: MockLatencyModel | None = None,
    ) -> MockBackend:
        return cls(
            model_id=model.id,
            replica_id=f"{model.id}-{replica_index}",
            tier=model.tier,
            max_concurrency=model.max_concurrency,
            latency_model=latency_model or latency_model_for_tier(model.tier),
        )

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def has_capacity(self) -> bool:
        return self._healthy and self._in_flight < self.max_concurrency

    def set_healthy(self, healthy: bool) -> None:
        self._healthy = healthy

    async def health(self) -> bool:
        return self._healthy

    async def generate(self, request: GenerationRequest) -> AsyncIterator[Token]:
        self._in_flight += 1
        completion_tokens = 0
        try:
            await asyncio.sleep(
                self.latency_model.sample_ttft(request.prompt_tokens, self._in_flight)
            )
            for index in range(request.max_tokens):
                await asyncio.sleep(self.latency_model.sample_itl(self._in_flight))
                completion_tokens = index + 1
                yield Token(
                    text=self._token_text(index),
                    index=index,
                    model_id=self.model_id,
                    replica_id=self.replica_id,
                    request_id=request.request_id,
                    is_final=index == request.max_tokens - 1,
                    finish_reason="length" if index == request.max_tokens - 1 else None,
                    prompt_tokens=request.prompt_tokens,
                    completion_tokens=completion_tokens,
                )
        finally:
            self._in_flight -= 1

    def _token_text(self, index: int) -> str:
        return "mock" if index == 0 else f" token-{index}"


def latency_model_for_tier(tier: str) -> MockLatencyModel:
    if tier == "large":
        return MockLatencyModel(
            base_ttft_ms=45.0,
            prompt_ms_per_token=0.05,
            base_itl_ms=14.0,
            concurrency_penalty=0.45,
        )
    return MockLatencyModel()
