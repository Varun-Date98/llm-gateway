from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from gateway.schemas import Token

HTTP_REQUESTS_TOTAL = Counter(
    "gateway_http_requests_total",
    "HTTP requests handled by the gateway.",
    ["method", "path", "status"],
)

HTTP_REQUEST_LATENCY_SECONDS = Histogram(
    "gateway_http_request_latency_seconds",
    "HTTP request latency in seconds.",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)

ADMISSION_TOTAL = Counter(
    "gateway_admission_total",
    "Admission decisions made by the scheduler.",
    ["decision", "reason"],
)

QUEUE_DEPTH = Gauge(
    "gateway_queue_depth",
    "Requests waiting in the scheduler queue.",
)

IN_FLIGHT = Gauge(
    "gateway_in_flight",
    "Active requests assigned to a backend replica.",
    ["model", "replica", "tier"],
)

TTFT_SECONDS = Histogram(
    "gateway_ttft_seconds",
    "Time to first token in seconds.",
    ["model", "replica"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)

GENERATION_SECONDS = Histogram(
    "gateway_generation_seconds",
    "End-to-end backend generation time in seconds.",
    ["model", "replica"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)

TOKENS_TOTAL = Counter(
    "gateway_tokens_total",
    "Tokens emitted by backend replicas.",
    ["model", "replica", "type"],
)

COST_USD_TOTAL = Counter(
    "gateway_cost_usd_total",
    "Estimated model cost in US dollars.",
    ["model"],
)

PREFIX_CACHE_TOTAL = Counter(
    "gateway_prefix_cache_total",
    "Prefix cache routing decisions.",
    ["result"],
)


def record_http_request(method: str, path: str, status_code: int, duration_seconds: float) -> None:
    HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status=str(status_code)).inc()
    HTTP_REQUEST_LATENCY_SECONDS.labels(method=method, path=path).observe(duration_seconds)


def record_admission(decision: str, reason: str) -> None:
    ADMISSION_TOTAL.labels(decision=decision, reason=reason).inc()


def set_queue_depth(depth: int) -> None:
    QUEUE_DEPTH.set(depth)


def set_in_flight(model_id: str, replica_id: str, tier: str, value: int) -> None:
    IN_FLIGHT.labels(model=model_id, replica=replica_id, tier=tier).set(value)


def record_first_token(token: Token, ttft_seconds: float) -> None:
    TTFT_SECONDS.labels(model=token.model_id, replica=token.replica_id).observe(ttft_seconds)


def record_generation(token: Token, duration_seconds: float) -> None:
    GENERATION_SECONDS.labels(model=token.model_id, replica=token.replica_id).observe(
        duration_seconds
    )


def record_token(token: Token) -> None:
    TOKENS_TOTAL.labels(model=token.model_id, replica=token.replica_id, type="completion").inc()


def record_prompt_tokens(model_id: str, replica_id: str, count: int) -> None:
    if count > 0:
        TOKENS_TOTAL.labels(model=model_id, replica=replica_id, type="prompt").inc(count)


def record_cost(model_id: str, cost_usd: float) -> None:
    if cost_usd > 0:
        COST_USD_TOTAL.labels(model=model_id).inc(cost_usd)


def record_prefix_cache(result: str) -> None:
    PREFIX_CACHE_TOTAL.labels(result=result).inc()


def render_metrics() -> bytes:
    return generate_latest()


def metrics_content_type() -> str:
    return CONTENT_TYPE_LATEST
