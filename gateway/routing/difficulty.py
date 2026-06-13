from __future__ import annotations

from dataclasses import dataclass, field

from gateway.backends.base import Backend
from gateway.backends.registry import NoHealthyReplicaError, ReplicaPool
from gateway.config import DifficultyRoutingConfig
from gateway.routing.classifier import HeuristicDifficultyClassifier
from gateway.routing.prefix_aware import PrefixAwareRouter
from gateway.schemas import GenerationRequest


@dataclass
class DifficultyRouter:
    """Route easy requests to cheaper tiers and hard requests to stronger tiers."""

    threshold: float = 0.6
    small_tier: str = "small"
    large_tier: str = "large"
    classifier: HeuristicDifficultyClassifier = field(
        default_factory=HeuristicDifficultyClassifier
    )
    within_tier_router: PrefixAwareRouter | None = None

    @classmethod
    def from_config(
        cls,
        config: DifficultyRoutingConfig,
        *,
        within_tier_router: PrefixAwareRouter | None = None,
    ) -> DifficultyRouter:
        if config.strategy != "heuristic":
            raise NotImplementedError("only heuristic difficulty routing is implemented")
        return cls(threshold=config.threshold, within_tier_router=within_tier_router)

    def score(self, request: GenerationRequest) -> float:
        return self.classifier.score(request)

    def select_tier(self, request: GenerationRequest, pool: ReplicaPool) -> str:
        preferred = self.large_tier if self.score(request) >= self.threshold else self.small_tier
        if preferred in pool.tiers:
            return preferred
        if self.small_tier in pool.tiers:
            return self.small_tier
        raise NoHealthyReplicaError(f"no configured tier for request {request.request_id}")

    def select(self, request: GenerationRequest, pool: ReplicaPool) -> Backend:
        tier = self.select_tier(request, pool)
        if self.within_tier_router is not None:
            return self.within_tier_router.select(request, pool, tier=tier)
        return pool.least_loaded(tier)
