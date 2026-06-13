from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from gateway.schemas import GenerationRequest, Token


@runtime_checkable
class Backend(Protocol):
    """Common shape shared by mock and real vLLM backends."""

    model_id: str
    replica_id: str
    tier: str

    def generate(self, request: GenerationRequest) -> AsyncIterator[Token]:
        """Stream generated tokens for a request."""
        ...

    async def health(self) -> bool:
        """Return whether this replica can currently receive work."""
        ...

    @property
    def in_flight(self) -> int:
        """Active requests currently assigned to this backend."""
        ...
