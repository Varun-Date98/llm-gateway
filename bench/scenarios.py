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
    stream: bool = True
    priority: Literal["interactive", "batch"] = "interactive"
    sweep: str = "default"
    policy: str = "smart"
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
            "metadata": {
                "scenario": self.name,
                "request_index": index,
                "sweep": self.sweep,
                "policy": self.policy,
                "workload": self.workload,
            },
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
        scenario
        for sweep, workload, max_tokens in [
            ("short_streaming", "short", 8),
            ("shared_prefix_streaming", "shared_prefix", 8),
            ("mixed_streaming", "mixed", 12),
        ]
        for scenario in streaming_sweep(sweep, workload, max_tokens=max_tokens)
    ]


def streaming_sweep(
    sweep: str,
    workload: WorkloadKind,
    *,
    max_tokens: int,
    policy: str = "smart",
    rates: tuple[int, ...] = (1, 2, 4, 8, 12),
    duration_seconds: float = 5,
) -> list[Scenario]:
    return [
        Scenario(
            name=f"{sweep}_{rate}rps",
            arrival_rate_rps=rate,
            duration_seconds=duration_seconds,
            workload=workload,
            max_tokens=max_tokens,
            stream=True,
            sweep=sweep,
            policy=policy,
            tags=(sweep, workload, "streaming"),
        )
        for rate in rates
    ]


def scenario_by_name(name: str) -> Scenario:
    for scenario in default_scenarios():
        if scenario.name == name:
            return scenario
    raise KeyError(name)
