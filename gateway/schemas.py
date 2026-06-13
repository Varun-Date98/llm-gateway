from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class RequestPriority(StrEnum):
    """Priority classes used by admission control and scheduling."""

    INTERACTIVE = "interactive"
    BATCH = "batch"


class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    """Subset of the OpenAI chat completion request the gateway needs to route."""

    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False
    stop: str | list[str] | None = None
    user: str | None = None
    priority: RequestPriority = RequestPriority.INTERACTIVE
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, messages: list[ChatMessage]) -> list[ChatMessage]:
        if not messages:
            raise ValueError("messages must contain at least one chat message")
        return messages

    def to_generation_request(self, *, request_id: str | None = None) -> GenerationRequest:
        return GenerationRequest(
            request_id=request_id or f"req_{uuid.uuid4().hex}",
            requested_model=self.model,
            messages=self.messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=self.stream,
            stop=self.stop,
            user=self.user,
            priority=self.priority,
            metadata=self.metadata,
            raw_request=self.model_dump(mode="json"),
        )


class GenerationRequest(BaseModel):
    """Internal request contract shared by routing, scheduling, and backends."""

    request_id: str
    requested_model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(ge=1)
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(gt=0.0, le=1.0)
    stream: bool = False
    stop: str | list[str] | None = None
    user: str | None = None
    priority: RequestPriority = RequestPriority.INTERACTIVE
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_request: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.monotonic)

    model_config = ConfigDict(extra="forbid")

    @computed_field
    @property
    def prompt_text(self) -> str:
        return "\n".join(f"{message.role}: {message.content}" for message in self.messages)

    @computed_field
    @property
    def prompt_tokens(self) -> int:
        # Cheap tokenizer approximation for routing. Backend-specific accounting comes later.
        text = self.prompt_text.strip()
        if not text:
            return 0
        return max(1, len(text) // 4)

    @computed_field
    @property
    def total_token_budget(self) -> int:
        return self.prompt_tokens + self.max_tokens


class Token(BaseModel):
    """Streaming unit emitted by any backend."""

    text: str
    index: int
    model_id: str
    replica_id: str
    request_id: str
    is_final: bool = False
    finish_reason: Literal["stop", "length", "error"] | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    generated_at: float = Field(default_factory=time.monotonic)


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length", "error"] | None = "stop"


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ChatCompletionChunkDelta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Literal["stop", "length", "error"] | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChunkChoice]
