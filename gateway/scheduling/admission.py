from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from gateway.config import AdmissionConfig


class AdmissionDecision(StrEnum):
    ACCEPT = "accept"
    QUEUE = "queue"
    SHED = "shed"


@dataclass(frozen=True)
class AdmissionResult:
    decision: AdmissionDecision
    reason: str


class AdmissionController:
    """Backpressure policy for accepting, queueing, or shedding requests."""

    def __init__(
        self,
        *,
        max_queue_depth: int,
        target_p99_ms: int,
        shed_when_queue_full: bool = True,
    ) -> None:
        self.max_queue_depth = max_queue_depth
        self.target_p99_ms = target_p99_ms
        self.shed_when_queue_full = shed_when_queue_full

    @classmethod
    def from_config(cls, config: AdmissionConfig) -> AdmissionController:
        return cls(
            max_queue_depth=config.max_queue_depth,
            target_p99_ms=config.target_p99_ms,
            shed_when_queue_full=config.shed_when_queue_full,
        )

    def decide(
        self,
        *,
        queue_depth: int,
        has_capacity: bool,
        estimated_wait_ms: float = 0.0,
    ) -> AdmissionResult:
        if has_capacity and queue_depth == 0:
            return AdmissionResult(AdmissionDecision.ACCEPT, "capacity_available")

        if queue_depth >= self.max_queue_depth and self.shed_when_queue_full:
            return AdmissionResult(AdmissionDecision.SHED, "queue_full")

        if estimated_wait_ms > self.target_p99_ms:
            return AdmissionResult(AdmissionDecision.SHED, "slo_risk")

        return AdmissionResult(AdmissionDecision.QUEUE, "queued_for_capacity")
