import json
from pathlib import Path

import yaml


def test_prometheus_config_parses() -> None:
    config = yaml.safe_load(Path("config/prometheus.yml").read_text(encoding="utf-8"))

    assert config["scrape_configs"][0]["job_name"] == "llm-gateway"
    assert config["scrape_configs"][0]["static_configs"][0]["targets"] == ["gateway:8000"]


def test_grafana_dashboard_parses() -> None:
    dashboard = json.loads(
        Path("config/grafana/dashboards/llm-gateway.json").read_text(encoding="utf-8")
    )

    assert dashboard["title"] == "LLM Gateway"
    assert len(dashboard["panels"]) == 4


def test_docker_compose_defines_observability_services() -> None:
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))

    assert set(compose["services"]) == {"gateway", "prometheus", "grafana"}
    assert compose["services"]["prometheus"]["ports"] == ["9090:9090"]
    assert compose["services"]["grafana"]["ports"] == ["3000:3000"]
