from pathlib import Path

import pytest
from pydantic import ValidationError

from gateway.config import GatewayConfig, load_config


def test_load_default_config() -> None:
    config = load_config()

    assert set(config.models) == {"small", "large"}
    assert config.models["small"].replicas == 2
    assert config.routing.within_tier == "prefix_aware"
    assert config.scheduling.priorities["interactive"] < config.scheduling.priorities["batch"]


def test_config_rejects_missing_small_tier() -> None:
    with pytest.raises(ValidationError, match="small-tier"):
        GatewayConfig.model_validate(
            {
                "models": {
                    "large": {
                        "id": "mock-large",
                        "tier": "large",
                        "backend": "mock",
                    }
                }
            }
        )


def test_vllm_model_requires_endpoint() -> None:
    with pytest.raises(ValidationError, match="endpoint"):
        GatewayConfig.model_validate(
            {
                "models": {
                    "small": {
                        "id": "llama-1b",
                        "tier": "small",
                        "backend": "vllm",
                    }
                }
            }
        )


def test_load_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "gateway.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(config_path)
