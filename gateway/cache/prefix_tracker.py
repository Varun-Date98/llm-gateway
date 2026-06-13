from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from gateway.backends.base import Backend
from gateway.schemas import GenerationRequest


@dataclass(frozen=True)
class PrefixEntry:
    prefix_hash: str
    replica_id: str
    tier: str
    last_seen: float


class PrefixTracker:
    """In-memory prompt-prefix affinity map with TTL and LRU eviction."""

    def __init__(
        self,
        *,
        ttl_seconds: int = 300,
        max_entries: int = 10000,
        prefix_chars: int = 2048,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.prefix_chars = prefix_chars
        self._clock = clock or time.monotonic
        self._entries: OrderedDict[str, PrefixEntry] = OrderedDict()

    def lookup(self, request: GenerationRequest) -> PrefixEntry | None:
        self._expire()
        prefix_hash = self.hash_request(request)
        entry = self._entries.get(prefix_hash)
        if entry is None:
            return None
        if self._is_expired(entry):
            self._entries.pop(prefix_hash, None)
            return None
        self._entries.move_to_end(prefix_hash)
        return entry

    def update(self, request: GenerationRequest, backend: Backend) -> PrefixEntry:
        self._expire()
        prefix_hash = self.hash_request(request)
        entry = PrefixEntry(
            prefix_hash=prefix_hash,
            replica_id=backend.replica_id,
            tier=backend.tier,
            last_seen=self._clock(),
        )
        self._entries[prefix_hash] = entry
        self._entries.move_to_end(prefix_hash)
        self._evict_lru()
        return entry

    def remove(self, prefix_hash: str) -> None:
        self._entries.pop(prefix_hash, None)

    def clear(self) -> None:
        self._entries.clear()

    def hash_request(self, request: GenerationRequest) -> str:
        prefix = request.prompt_text[: self.prefix_chars]
        return hashlib.sha256(prefix.encode("utf-8")).hexdigest()

    def __len__(self) -> int:
        self._expire()
        return len(self._entries)

    def _expire(self) -> None:
        expired = [
            prefix_hash
            for prefix_hash, entry in self._entries.items()
            if self._is_expired(entry)
        ]
        for prefix_hash in expired:
            self._entries.pop(prefix_hash, None)

    def _evict_lru(self) -> None:
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def _is_expired(self, entry: PrefixEntry) -> bool:
        return self._clock() - entry.last_seen > self.ttl_seconds
