from collections.abc import AsyncIterator

from gateway.backends.base import Backend
from gateway.schemas import ChatCompletionRequest, GenerationRequest, RequestPriority, Token


def test_chat_completion_request_converts_to_generation_request() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "auto",
            "messages": [
                {"role": "system", "content": "Answer briefly."},
                {"role": "user", "content": "What is an inference gateway?"},
            ],
            "max_tokens": 32,
            "priority": "batch",
        }
    )

    generation_request = request.to_generation_request(request_id="req_test")

    assert generation_request.request_id == "req_test"
    assert generation_request.requested_model == "auto"
    assert generation_request.priority == RequestPriority.BATCH
    assert generation_request.prompt_tokens > 0
    assert generation_request.total_token_budget == generation_request.prompt_tokens + 32


def test_backend_protocol_is_runtime_checkable() -> None:
    class MinimalBackend:
        model_id = "mock-small"
        replica_id = "mock-small-0"
        tier = "small"

        async def generate(self, request: GenerationRequest) -> AsyncIterator[Token]:
            yield Token(
                text="ok",
                index=0,
                model_id=self.model_id,
                replica_id=self.replica_id,
                request_id=request.request_id,
            )

        async def health(self) -> bool:
            return True

        def reserve(self) -> bool:
            return True

        def release(self) -> None:
            return None

        @property
        def in_flight(self) -> int:
            return 0

    assert isinstance(MinimalBackend(), Backend)
