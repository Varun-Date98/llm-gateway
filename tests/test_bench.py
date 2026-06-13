from pathlib import Path

import pytest

from bench.loadgen import estimate_prompt_tokens, run_open_loop
from bench.report import generate_report, summarize
from bench.run import benchmark_client, run_benchmark
from bench.scenarios import Scenario, default_scenarios, scenario_by_name


def test_default_scenarios_are_lookupable() -> None:
    scenarios = default_scenarios()

    assert scenarios
    assert scenario_by_name(scenarios[0].name) == scenarios[0]
    assert scenarios[0].request_count() > 0
    assert all(scenario.stream for scenario in scenarios)
    assert {scenario.arrival_rate_rps for scenario in scenarios} == {1, 2, 4, 8, 12}
    assert {scenario.sweep for scenario in scenarios} == {
        "short_streaming",
        "shared_prefix_streaming",
        "mixed_streaming",
    }


def test_estimate_prompt_tokens_matches_gateway_style() -> None:
    tokens = estimate_prompt_tokens([{"role": "user", "content": "hello there"}])

    assert tokens > 0


@pytest.mark.asyncio
async def test_run_open_loop_against_in_process_gateway() -> None:
    scenario = Scenario(
        name="test_one_request",
        arrival_rate_rps=1,
        duration_seconds=0.1,
        workload="short",
        max_tokens=1,
    )

    async with benchmark_client() as client:
        result = await run_open_loop(client, scenario)

    assert result.scenario["name"] == "test_one_request"
    assert len(result.requests) == 1
    assert result.requests[0].status_code == 200
    assert result.requests[0].completion_tokens == 1


@pytest.mark.asyncio
async def test_run_benchmark_returns_json_ready_results() -> None:
    scenario = Scenario(
        name="test_benchmark",
        arrival_rate_rps=1,
        duration_seconds=0.1,
        workload="shared_prefix",
        max_tokens=1,
    )

    results = await run_benchmark([scenario])

    assert results["metadata"]["base_url"] == "in-process"
    assert results["scenarios"][0]["scenario"]["name"] == "test_benchmark"
    assert results["scenarios"][0]["requests"][0]["status_code"] == 200


def test_report_generation_writes_summary_files(tmp_path: Path) -> None:
    input_path = tmp_path / "benchmark.json"
    input_path.write_text(
        """
{
  "metadata": {"base_url": "test"},
  "scenarios": [
    {
      "scenario": {
        "name": "sample",
        "sweep": "sample_sweep",
        "policy": "smart",
        "workload": "short",
        "arrival_rate_rps": 1
      },
      "requests": [
        {
          "scenario": "sample",
          "request_id": "sample-0",
          "status_code": 200,
          "latency_seconds": 0.1,
          "ttft_seconds": 0.1,
          "model": "mock-small",
          "prompt_tokens": 10,
          "completion_tokens": 2,
          "error": null
        }
      ]
    }
  ]
}
""",
        encoding="utf-8",
    )

    outputs = generate_report(input_path, tmp_path)
    summaries = summarize(
        {
            "scenarios": [
                {
                    "scenario": {
                        "name": "sample",
                        "sweep": "sample_sweep",
                        "policy": "smart",
                        "workload": "short",
                        "arrival_rate_rps": 1,
                    },
                    "requests": [
                        {
                            "status_code": 200,
                            "latency_seconds": 0.1,
                            "ttft_seconds": 0.1,
                            "model": "mock-small",
                            "prompt_tokens": 10,
                            "completion_tokens": 2,
                        }
                    ],
                }
            ]
        }
    )

    assert Path(outputs["summary_json"]).exists()
    assert Path(outputs["summary_markdown"]).exists()
    assert "sample_sweep" in Path(outputs["summary_markdown"]).read_text(encoding="utf-8")
    assert summaries[0]["p99_latency_s"] == 0.1
    assert summaries[0]["p99_ttft_s"] == 0.1
    assert summaries[0]["sweep"] == "sample_sweep"
