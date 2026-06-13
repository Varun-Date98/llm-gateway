from __future__ import annotations

import asyncio
import inspect
from collections import defaultdict
from collections.abc import Iterable

from gateway.backends.base import Backend
from gateway.backends.mock import MockBackend
from gateway.backends.vllm import VllmBackend
from gateway.config import GatewayConfig


class NoHealthyReplicaError(RuntimeError):
    """Raised when a routing policy asks for a tier with no available replica."""


class ReplicaPool:
    """Fleet view used by routers and schedulers."""

    def __init__(self, replicas: Iterable[Backend]) -> None:
        self._replicas = list(replicas)
        self._by_tier: dict[str, list[Backend]] = defaultdict(list)
        self._health: dict[str, bool] = {}
        for replica in self._replicas:
            self._by_tier[replica.tier].append(replica)
            self._health[replica.replica_id] = True

    @classmethod
    def from_config(cls, config: GatewayConfig) -> ReplicaPool:
        replicas: list[Backend] = []
        for model in config.models.values():
            for replica_index in range(model.replicas):
                if model.backend == "mock":
                    replicas.append(
                        MockBackend.from_model_config(model, replica_index=replica_index)
                    )
                elif model.backend == "vllm":
                    replicas.append(
                        VllmBackend.from_model_config(model, replica_index=replica_index)
                    )
                else:
                    raise NotImplementedError(f"unsupported backend: {model.backend}")
        return cls(replicas)

    @property
    def replicas(self) -> list[Backend]:
        return list(self._replicas)

    @property
    def tiers(self) -> set[str]:
        return set(self._by_tier)

    def get(self, replica_id: str) -> Backend:
        for replica in self._replicas:
            if replica.replica_id == replica_id:
                return replica
        raise KeyError(replica_id)

    def replicas_for_tier(self, tier: str) -> list[Backend]:
        return list(self._by_tier.get(tier, []))

    def healthy_replicas(self, tier: str | None = None) -> list[Backend]:
        candidates = self.replicas_for_tier(tier) if tier is not None else self._replicas
        return [
            replica
            for replica in candidates
            if self._health.get(replica.replica_id, False)
            and getattr(replica, "has_capacity", True)
        ]

    def least_loaded(
        self,
        tier: str | None = None,
        candidates: Iterable[Backend] | None = None,
    ) -> Backend:
        eligible = list(candidates) if candidates is not None else self.healthy_replicas(tier)
        eligible = [
            replica
            for replica in eligible
            if self._health.get(replica.replica_id, False)
            and getattr(replica, "has_capacity", True)
        ]
        if not eligible:
            label = tier or "any tier"
            raise NoHealthyReplicaError(f"no healthy replicas available for {label}")
        return min(eligible, key=lambda replica: (replica.in_flight, replica.replica_id))

    def mark_health(self, replica_id: str, healthy: bool) -> None:
        self.get(replica_id)
        self._health[replica_id] = healthy

    def is_healthy(self, replica_id: str) -> bool:
        return self._health.get(replica_id, False)

    async def refresh_health(self) -> dict[str, bool]:
        results = await asyncio.gather(
            *(replica.health() for replica in self._replicas),
            return_exceptions=True,
        )
        for replica, result in zip(self._replicas, results, strict=True):
            self._health[replica.replica_id] = (
                bool(result) if not isinstance(result, Exception) else False
            )
        return dict(self._health)

    def in_flight_by_replica(self) -> dict[str, int]:
        return {replica.replica_id: replica.in_flight for replica in self._replicas}

    def in_flight_by_tier(self) -> dict[str, int]:
        return {
            tier: sum(replica.in_flight for replica in replicas)
            for tier, replicas in self._by_tier.items()
        }

    async def close(self) -> None:
        close_results = []
        for replica in self._replicas:
            close = getattr(replica, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    close_results.append(result)
        if close_results:
            await asyncio.gather(*close_results, return_exceptions=True)
