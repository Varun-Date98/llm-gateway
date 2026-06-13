from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from fastapi import FastAPI, Request

from gateway.app import register_routes
from gateway.backends.registry import ReplicaPool
from gateway.cache.prefix_tracker import PrefixTracker
from gateway.config import GatewayConfig, load_config
from gateway.observability import metrics
from gateway.routing.difficulty import DifficultyRouter
from gateway.routing.prefix_aware import PrefixAwareRouter
from gateway.scheduling.admission import AdmissionController
from gateway.scheduling.queue import PriorityRequestQueue
from gateway.scheduling.scheduler import Scheduler


@dataclass
class GatewayRuntime:
    config: GatewayConfig
    pool: ReplicaPool
    prefix_tracker: PrefixTracker
    router: DifficultyRouter
    admission: AdmissionController
    queue: PriorityRequestQueue
    scheduler: Scheduler


def build_runtime(config_path: str | Path | None = None) -> GatewayRuntime:
    config = load_config(config_path)
    pool = ReplicaPool.from_config(config)
    prefix_tracker = PrefixTracker(
        ttl_seconds=config.prefix_cache.ttl_seconds,
        max_entries=config.prefix_cache.max_entries,
        prefix_chars=config.prefix_cache.prefix_chars,
    )
    within_tier_router = build_within_tier_router(config, prefix_tracker)
    router = DifficultyRouter.from_config(
        config.routing.difficulty,
        within_tier_router=within_tier_router,
    )
    admission = AdmissionController.from_config(config.admission)
    queue = PriorityRequestQueue(config.scheduling.priorities)
    scheduler = Scheduler(
        pool=pool,
        router=router,
        admission=admission,
        queue=queue,
        dispatch_interval_ms=config.scheduling.dispatch_interval_ms,
    )
    return GatewayRuntime(
        config=config,
        pool=pool,
        prefix_tracker=prefix_tracker,
        router=router,
        admission=admission,
        queue=queue,
        scheduler=scheduler,
    )


def build_within_tier_router(
    config: GatewayConfig,
    prefix_tracker: PrefixTracker,
) -> PrefixAwareRouter | None:
    strategy = config.routing.within_tier
    if strategy == "prefix_aware":
        return PrefixAwareRouter(
            tracker=prefix_tracker,
            max_load_ratio=config.routing.prefix_affinity_max_load_ratio,
        )
    if strategy == "least_loaded":
        return None
    raise NotImplementedError(f"within-tier routing strategy is not implemented: {strategy}")


def create_app(config_path: str | Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime = build_runtime(config_path)
        app.state.gateway = runtime
        runtime.scheduler.start()
        try:
            yield
        finally:
            await runtime.scheduler.stop(drain=True)

    app = FastAPI(title="LLM Gateway", version="0.1.0", lifespan=lifespan)
    register_routes(app)
    register_metrics_middleware(app)
    return app


def register_metrics_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def record_http_metrics(request: Request, call_next):
        started = monotonic()
        response = await call_next(request)
        metrics.record_http_request(
            request.method,
            request.url.path,
            response.status_code,
            monotonic() - started,
        )
        return response


app = create_app()
