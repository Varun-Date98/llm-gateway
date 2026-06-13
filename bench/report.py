from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from gateway.config import load_config


def load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(results: dict[str, Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for scenario in results.get("scenarios", []):
        requests = scenario.get("requests", [])
        successes = [request for request in requests if request.get("status_code") == 200]
        latencies = [request["latency_seconds"] for request in successes]
        ttfts = [
            request["ttft_seconds"]
            for request in successes
            if request.get("ttft_seconds") is not None
        ]
        prompt_tokens = sum(int(request.get("prompt_tokens") or 0) for request in successes)
        completion_tokens = sum(
            int(request.get("completion_tokens") or 0) for request in successes
        )
        cost = estimate_cost(successes)
        scenario_config = scenario.get("scenario", {})
        summaries.append(
            {
                "scenario": scenario_config.get("name", "unknown"),
                "sweep": scenario_config.get("sweep", "default"),
                "policy": scenario_config.get("policy", "smart"),
                "workload": scenario_config.get("workload", "unknown"),
                "arrival_rate_rps": scenario_config.get("arrival_rate_rps", 0),
                "requests": len(requests),
                "successes": len(successes),
                "errors": len(requests) - len(successes),
                "p50_latency_s": percentile(latencies, 50),
                "p95_latency_s": percentile(latencies, 95),
                "p99_latency_s": percentile(latencies, 99),
                "p99_ttft_s": percentile(ttfts, 99),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "estimated_cost_usd": cost,
                "cost_per_1m_tokens_usd": cost_per_1m(cost, prompt_tokens + completion_tokens),
            }
        )
    return summaries


def percentile(values: list[float], pct: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil((pct / 100) * len(ordered)) - 1
    return ordered[max(0, min(rank, len(ordered) - 1))]


def estimate_cost(requests: list[dict[str, Any]]) -> float:
    config = load_config()
    models = {model.id: model for model in config.models.values()}
    total = 0.0
    for request in requests:
        model_id = request.get("model")
        model = models.get(model_id)
        if model is None:
            continue
        total += int(request.get("prompt_tokens") or 0) * model.input_cost_per_1m / 1_000_000
        total += (
            int(request.get("completion_tokens") or 0)
            * model.output_cost_per_1m
            / 1_000_000
        )
    return total


def cost_per_1m(cost: float, tokens: int) -> float | None:
    if tokens <= 0:
        return None
    return cost / tokens * 1_000_000


def write_markdown_summary(summaries: list[dict[str, Any]], output_path: Path) -> None:
    lines = [
        "# Benchmark Summary",
        "",
        (
            "| Sweep | RPS | Requests | Successes | P99 latency (s) | "
            "P99 TTFT (s) | Cost / 1M tokens |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        row = (
            "| {sweep} | {rps} | {requests} | {successes} | "
            "{p99_latency_s} | {p99_ttft_s} | {cost} |"
        )
        lines.append(
            row.format(
                sweep=summary["sweep"],
                rps=format_number(summary["arrival_rate_rps"]),
                requests=summary["requests"],
                successes=summary["successes"],
                p99_latency_s=format_number(summary["p99_latency_s"]),
                p99_ttft_s=format_number(summary["p99_ttft_s"]),
                cost=format_number(summary["cost_per_1m_tokens_usd"]),
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(summaries: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return []

    written: list[Path] = []
    by_sweep = group_by_sweep(summaries)

    plt.figure(figsize=(10, 5))
    for sweep, rows in by_sweep.items():
        ordered = sorted(rows, key=lambda row: row["arrival_rate_rps"])
        plt.plot(
            [row["arrival_rate_rps"] for row in ordered],
            [row["p99_latency_s"] or 0 for row in ordered],
            marker="o",
            label=sweep,
        )
    plt.xlabel("Offered load (RPS)")
    plt.ylabel("P99 latency (s)")
    plt.legend()
    plt.tight_layout()
    latency_path = output_dir / "p99_latency_vs_load.png"
    plt.savefig(latency_path)
    plt.close()
    written.append(latency_path)

    plt.figure(figsize=(10, 5))
    for sweep, rows in by_sweep.items():
        ordered = sorted(rows, key=lambda row: row["arrival_rate_rps"])
        plt.plot(
            [row["arrival_rate_rps"] for row in ordered],
            [row["p99_ttft_s"] or 0 for row in ordered],
            marker="o",
            label=sweep,
        )
    plt.xlabel("Offered load (RPS)")
    plt.ylabel("P99 TTFT (s)")
    plt.legend()
    plt.tight_layout()
    ttft_path = output_dir / "p99_ttft_vs_load.png"
    plt.savefig(ttft_path)
    plt.close()
    written.append(ttft_path)

    plt.figure(figsize=(10, 5))
    labels = [f"{summary['sweep']} @ {summary['arrival_rate_rps']}rps" for summary in summaries]
    costs = [summary["cost_per_1m_tokens_usd"] or 0 for summary in summaries]
    plt.bar(labels, costs)
    plt.ylabel("Estimated cost per 1M tokens (USD)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    cost_path = output_dir / "cost_per_1m_tokens.png"
    plt.savefig(cost_path)
    plt.close()
    written.append(cost_path)
    return written


def write_summary_json(summaries: list[dict[str, Any]], output_path: Path) -> None:
    output_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")


def generate_report(input_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = summarize(load_results(input_path))
    summary_json = output_dir / "benchmark_summary.json"
    summary_md = output_dir / "benchmark_summary.md"
    write_summary_json(summaries, summary_json)
    write_markdown_summary(summaries, summary_md)
    plots = write_plots(summaries, output_dir)
    return {
        "summary_json": str(summary_json),
        "summary_markdown": str(summary_md),
        "plots": [str(path) for path in plots],
    }


def format_number(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def group_by_model(requests: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for request in requests:
        counts[request.get("model") or "unknown"] += 1
    return dict(counts)


def group_by_sweep(summaries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        grouped[summary["sweep"]].append(summary)
    return dict(grouped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reports from benchmark JSON.")
    parser.add_argument("input", type=Path, help="Benchmark JSON from bench.run.")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate_report(args.input, args.out_dir)
    for output in [outputs["summary_json"], outputs["summary_markdown"], *outputs["plots"]]:
        print(output)


if __name__ == "__main__":
    main()
