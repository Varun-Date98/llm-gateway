from __future__ import annotations

from typing import Protocol

from gateway.backends.base import Backend
from gateway.backends.registry import ReplicaPool
from gateway.schemas import GenerationRequest


class Router(Protocol):
    """Strategy that chooses one backend replica for a request."""

    def select(self, request: GenerationRequest, pool: ReplicaPool) -> Backend:
        ...


class TierRouter(Protocol):
    """Strategy that chooses a model tier before replica selection."""

    def select_tier(self, request: GenerationRequest, pool: ReplicaPool) -> str:
        ...
