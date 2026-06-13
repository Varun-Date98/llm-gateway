from __future__ import annotations

import argparse
import json
import os
from collections.abc import AsyncIterator
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from gateway.observability import metrics
from gateway.scheduling.scheduler import AdmissionRejected, ScheduledStream, SchedulerClosed
from gateway.schemas import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    GenerationRequest,
    Token,
)


def register_routes(app: FastAPI) -> None:
    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        runtime = get_runtime(request)
        healthy = len(runtime.pool.healthy_replicas())
        return {
            "status": "ok",
            "scheduler_running": runtime.scheduler.is_running,
            "replicas": len(runtime.pool.replicas),
            "healthy_replicas": healthy,
        }

    @app.get("/v1/models")
    async def list_models(request: Request) -> dict[str, Any]:
        runtime = get_runtime(request)
        models = [
            {
                "id": model.id,
                "object": "model",
                "owned_by": "llm-gateway",
                "tier": model.tier,
                "replicas": model.replicas,
            }
            for model in runtime.config.models.values()
        ]
        return {"object": "list", "data": models}

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics(request: Request) -> Response:
        runtime = get_runtime(request)
        if not runtime.config.metrics.enabled:
            raise HTTPException(status_code=404, detail="metrics disabled")
        return Response(
            content=metrics.render_metrics(),
            media_type=metrics.metrics_content_type(),
        )

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        request: Request,
        body: ChatCompletionRequest,
    ) -> ChatCompletionResponse | StreamingResponse:
        runtime = get_runtime(request)
        generation_request = body.to_generation_request(
            request_id=request.headers.get("x-request-id")
        )
        try:
            stream = await runtime.scheduler.submit(generation_request)
        except AdmissionRejected as error:
            raise HTTPException(status_code=429, detail=error.result.reason) from error
        except SchedulerClosed as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

        if body.stream:
            return StreamingResponse(
                stream_chat_completion(generation_request, stream, runtime),
                media_type="text/event-stream",
            )
        return await collect_chat_completion(generation_request, stream, runtime)


def get_runtime(request: Request):
    return request.app.state.gateway


async def collect_chat_completion(
    request: GenerationRequest,
    stream: ScheduledStream,
    runtime,
) -> ChatCompletionResponse:
    parts: list[str] = []
    final_token: Token | None = None
    async for token in stream:
        final_token = token
        parts.append(token.text)

    model_id = (
        final_token.model_id
        if final_token is not None
        else request.requested_model or "unknown"
    )
    completion_tokens = final_token.completion_tokens if final_token is not None else 0
    finish_reason = final_token.finish_reason if final_token is not None else "stop"
    metrics.record_cost(
        model_id,
        estimate_cost(runtime, model_id, request.prompt_tokens, completion_tokens or 0),
    )
    return ChatCompletionResponse(
        id=request.request_id,
        model=model_id,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content="".join(parts)),
                finish_reason=finish_reason,
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=request.prompt_tokens,
            completion_tokens=completion_tokens or 0,
            total_tokens=request.prompt_tokens + (completion_tokens or 0),
        ),
    )


async def stream_chat_completion(
    request: GenerationRequest,
    stream: ScheduledStream,
    runtime,
) -> AsyncIterator[str]:
    model_id = request.requested_model or "gateway"
    completion_tokens = 0
    async for token in stream:
        model_id = token.model_id
        completion_tokens = token.completion_tokens or completion_tokens
        chunk = ChatCompletionChunk(
            id=request.request_id,
            model=model_id,
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(content=token.text),
                    finish_reason=token.finish_reason if token.is_final else None,
                )
            ],
        )
        yield format_sse(chunk.model_dump(mode="json"))
    metrics.record_cost(
        model_id,
        estimate_cost(runtime, model_id, request.prompt_tokens, completion_tokens),
    )
    yield format_sse("[DONE]")


def format_sse(payload: dict[str, Any] | str) -> str:
    data = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {data}\n\n"


def estimate_cost(runtime, model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    model_config = next(
        (model for model in runtime.config.models.values() if model.id == model_id),
        None,
    )
    if model_config is None:
        return 0.0
    input_cost = prompt_tokens * model_config.input_cost_per_1m / 1_000_000
    output_cost = completion_tokens * model_config.output_cost_per_1m / 1_000_000
    return input_cost + output_cost


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LLM gateway API server.")
    parser.add_argument("--host", default=os.getenv("LLM_GATEWAY_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("LLM_GATEWAY_PORT", "8000")))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "gateway.server:create_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=True,
    )


if __name__ == "__main__":
    main()
