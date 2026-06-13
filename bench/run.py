from __future__ import annotations

import argparse
import asyncio
import json
import platform
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx

from bench.loadgen import ScenarioResult, run_open_loop
from bench.scenarios import Scenario, default_scenarios, scenario_by_name
from gateway.server import create_app


@asynccontextmanager
async def benchmark_client(base_url: str | None = None) -> AsyncIterator[httpx.AsyncClient]:
    if base_url is not None:
        async with httpx.AsyncClient(base_url=base_url, timeout=None) as client:
            yield client
        return

    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://llm-gateway.local",
            timeout=None,
        ) as client:
            yield client


async def run_benchmark(
    scenarios: list[Scenario],
    *,
    base_url: str | None = None,
) -> dict[str, Any]:
    started_at = time.time()
    scenario_results: list[ScenarioResult] = []
    async with benchmark_client(base_url) as client:
        for scenario in scenarios:
            scenario_results.append(await run_open_loop(client, scenario))

    return {
        "metadata": {
            "created_at": int(started_at),
            "base_url": base_url or "in-process",
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "scenarios": [result.to_dict() for result in scenario_results],
    }


def write_results(results: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM gateway benchmark scenarios.")
    parser.add_argument("--base-url", default=None, help="Gateway URL. Defaults to in-process app.")
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Scenario name to run. May be passed more than once. Defaults to all.",
    )
    parser.add_argument(
        "--out",
        default="results/mock_benchmark.json",
        help="Path for JSON benchmark results.",
    )
    parser.add_argument("--list", action="store_true", help="List available scenarios and exit.")
    return parser.parse_args()


def selected_scenarios(names: list[str]) -> list[Scenario]:
    if not names:
        return default_scenarios()
    return [scenario_by_name(name) for name in names]


def main() -> None:
    args = parse_args()
    if args.list:
        for scenario in default_scenarios():
            print(scenario.name)
        return

    results = asyncio.run(run_benchmark(selected_scenarios(args.scenario), base_url=args.base_url))
    output_path = Path(args.out)
    write_results(results, output_path)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
