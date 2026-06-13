from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

WorkloadKind = Literal["short", "long", "shared_prefix", "mixed"]


@dataclass(frozen=True)
class Scenario:
    """Declarative load-test scenario."""

    name: str
    arrival_rate_rps: float
    duration_seconds: float
    workload: WorkloadKind = "short"
    max_tokens: int = 16
    stream: bool = False
    priority: Literal["interactive", "batch"] = "interactive"
    max_concurrency: int | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    def request_count(self) -> int:
        return max(1, int(round(self.arrival_rate_rps * self.duration_seconds)))

    def payload(self, index: int) -> dict:
        return {
            "messages": [{"role": "user", "content": prompt_for(self.workload, index)}],
            "max_tokens": self.max_tokens,
            "stream": self.stream,
            "priority": self.priority,
            "metadata": {"scenario": self.name, "request_index": index},
        }


def prompt_for(workload: WorkloadKind, index: int) -> str:
    if workload == "short":
        return f"Summarize request {index} in one sentence."
    if workload == "long":
        return (
            "Analyze this Python service architecture, explain the routing tradeoffs, "
            "and identify failure modes. "
            * 80
        ) + f"\nRequest: {index}"
    if workload == "shared_prefix":
        return (
            "System context: You are evaluating the same customer support policy. "
            "Use the policy below for every answer. Policy: refunds require a receipt, "
            "manager approval above $200, and escalation for account security issues.\n"
            f"User case {index}: write a concise support reply."
        )
    if workload == "mixed":
        return prompt_for("long" if index % 5 == 0 else "short", index)
    raise ValueError(f"unknown workload: {workload}")


def default_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="mock_short_2rps",
            arrival_rate_rps=2,
            duration_seconds=5,
            workload="short",
            max_tokens=8,
            tags=("smoke", "short"),
        ),
        Scenario(
            name="mock_shared_prefix_4rps",
            arrival_rate_rps=4,
            duration_seconds=5,
            workload="shared_prefix",
            max_tokens=8,
            tags=("prefix", "smoke"),
        ),
        Scenario(
            name="mock_mixed_6rps",
            arrival_rate_rps=6,
            duration_seconds=5,
            workload="mixed",
            max_tokens=12,
            tags=("mixed", "load"),
        ),
    ]


def scenario_by_name(name: str) -> Scenario:
    for scenario in default_scenarios():
        if scenario.name == name:
            return scenario
    raise KeyError(name)
