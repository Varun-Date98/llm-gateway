from __future__ import annotations

from dataclasses import dataclass, field

from gateway.backends.base import Backend
from gateway.backends.registry import ReplicaPool
from gateway.cache.prefix_tracker import PrefixTracker
from gateway.observability import metrics
from gateway.schemas import GenerationRequest


@dataclass
class PrefixAwareRouter:
    """Prefer cache-affine replicas while bounding hot spots by load."""

    tracker: PrefixTracker = field(default_factory=PrefixTracker)
    max_load_ratio: float = 1.5

    def select(
        self,
        request: GenerationRequest,
        pool: ReplicaPool,
        *,
        tier: str | None = None,
    ) -> Backend:
        fallback = pool.least_loaded(tier)
        entry = self.tracker.lookup(request)
        if entry is None:
            metrics.record_prefix_cache("miss")
            return fallback
        if tier is not None and entry.tier != tier:
            metrics.record_prefix_cache("tier_mismatch")
            return fallback

        try:
            preferred = pool.get(entry.replica_id)
        except KeyError:
            self.tracker.remove(entry.prefix_hash)
            metrics.record_prefix_cache("stale")
            return fallback

        if preferred.tier != fallback.tier:
            metrics.record_prefix_cache("tier_mismatch")
            return fallback
        if not pool.is_healthy(preferred.replica_id):
            metrics.record_prefix_cache("unhealthy")
            return fallback
        if not getattr(preferred, "has_capacity", True):
            metrics.record_prefix_cache("full")
            return fallback
        if self._would_hotspot(preferred, fallback):
            metrics.record_prefix_cache("overloaded")
            return fallback
        metrics.record_prefix_cache("hit")
        return preferred

    def record(self, request: GenerationRequest, backend: Backend) -> None:
        self.tracker.update(request, backend)

    def _would_hotspot(self, preferred: Backend, fallback: Backend) -> bool:
        if preferred.replica_id == fallback.replica_id:
            return False
        if fallback.in_flight == 0:
            return preferred.in_flight > 0
        return preferred.in_flight / fallback.in_flight > self.max_load_ratio
