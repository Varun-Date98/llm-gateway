from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)

DEFAULT_CONFIG_PATH = Path("config/gateway.yaml")


class ModelConfig(BaseModel):
    id: str
    tier: str
    backend: Literal["mock", "vllm"] = "mock"
    replicas: int = Field(default=1, ge=1)
    endpoint: HttpUrl | None = None
    api_key_env: str | None = None
    request_timeout_seconds: float = Field(default=120.0, gt=0.0)
    health_path: str = "/health"
    max_concurrency: int = Field(default=8, ge=1)
    context_window: int = Field(default=8192, ge=1)
    input_cost_per_1m: float = Field(default=0.0, ge=0.0)
    output_cost_per_1m: float = Field(default=0.0, ge=0.0)
    quality_weight: float = Field(default=1.0, ge=0.0)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def require_vllm_endpoint(self) -> ModelConfig:
        if self.backend == "vllm" and self.endpoint is None:
            raise ValueError("vllm backends require an endpoint")
        return self

    @field_validator("health_path")
    @classmethod
    def validate_health_path(cls, path: str) -> str:
        if not path.startswith("/"):
            raise ValueError("health_path must start with '/'")
        return path


class DifficultyRoutingConfig(BaseModel):
    strategy: Literal["heuristic", "classifier"] = "heuristic"
    threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")


class RoutingConfig(BaseModel):
    difficulty: DifficultyRoutingConfig = Field(default_factory=DifficultyRoutingConfig)
    within_tier: Literal["prefix_aware", "least_loaded", "round_robin"] = "prefix_aware"
    prefix_affinity_max_load_ratio: float = Field(default=1.5, ge=1.0)

    model_config = ConfigDict(extra="forbid")


class AdmissionConfig(BaseModel):
    max_queue_depth: int = Field(default=256, ge=0)
    target_p99_ms: int = Field(default=2000, ge=1)
    shed_when_queue_full: bool = True
    default_service_time_ms: float = Field(default=250.0, gt=0.0)
    service_time_ewma_alpha: float = Field(default=0.2, gt=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")


class HealthConfig(BaseModel):
    refresh_interval_seconds: float = Field(default=5.0, gt=0.0)

    model_config = ConfigDict(extra="forbid")


class SchedulingConfig(BaseModel):
    priorities: dict[str, int] = Field(default_factory=lambda: {"interactive": 0, "batch": 1})
    dispatch_interval_ms: int = Field(default=5, ge=1)

    model_config = ConfigDict(extra="forbid")

    @field_validator("priorities")
    @classmethod
    def validate_priorities(cls, priorities: dict[str, int]) -> dict[str, int]:
        required = {"interactive", "batch"}
        missing = required.difference(priorities)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"priorities missing required classes: {names}")
        if any(value < 0 for value in priorities.values()):
            raise ValueError("priority values must be non-negative")
        return priorities


class PrefixCacheConfig(BaseModel):
    ttl_seconds: int = Field(default=300, ge=1)
    max_entries: int = Field(default=10000, ge=1)
    prefix_chars: int = Field(default=2048, ge=1)

    model_config = ConfigDict(extra="forbid")


class MetricsConfig(BaseModel):
    enabled: bool = True
    path: str = "/metrics"

    model_config = ConfigDict(extra="forbid")

    @field_validator("path")
    @classmethod
    def validate_path(cls, path: str) -> str:
        if not path.startswith("/"):
            raise ValueError("metrics path must start with '/'")
        return path


class GatewayConfig(BaseModel):
    models: dict[str, ModelConfig]
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    admission: AdmissionConfig = Field(default_factory=AdmissionConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)
    scheduling: SchedulingConfig = Field(default_factory=SchedulingConfig)
    prefix_cache: PrefixCacheConfig = Field(default_factory=PrefixCacheConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

    model_config = ConfigDict(extra="forbid")

    @field_validator("models")
    @classmethod
    def validate_models(cls, models: dict[str, ModelConfig]) -> dict[str, ModelConfig]:
        if not models:
            raise ValueError("at least one model must be configured")
        tiers = {model.tier for model in models.values()}
        if "small" not in tiers:
            raise ValueError("at least one small-tier model is required")
        return models

    def tiers(self) -> set[str]:
        return {model.tier for model in self.models.values()}


def load_config(path: str | Path | None = None) -> GatewayConfig:
    """Load and validate gateway policy from YAML."""

    config_path = Path(path or os.getenv("LLM_GATEWAY_CONFIG", DEFAULT_CONFIG_PATH))
    if not config_path.exists():
        raise FileNotFoundError(f"gateway config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    if not isinstance(raw_config, dict):
        raise ValueError(f"gateway config must be a YAML mapping: {config_path}")

    return GatewayConfig.model_validate(raw_config)


@lru_cache(maxsize=1)
def get_config(path: str | Path | None = None) -> GatewayConfig:
    """Cached config accessor for app startup code."""

    return load_config(path)
