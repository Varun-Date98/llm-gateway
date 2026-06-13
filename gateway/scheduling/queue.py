from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import count
from typing import Any

from gateway.schemas import GenerationRequest


@dataclass(order=True)
class QueuedRequest:
    priority: int
    sequence: int
    enqueued_at: float = field(compare=False)
    request: GenerationRequest = field(compare=False)
    payload: Any = field(compare=False, default=None)


class PriorityRequestQueue:
    """Async priority queue where lower priority numbers are dispatched first."""

    def __init__(
        self,
        priorities: dict[str, int] | None = None,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._priorities = priorities or {"interactive": 0, "batch": 1}
        self._clock = clock or time.monotonic
        self._sequence = count()
        self._queue: asyncio.PriorityQueue[QueuedRequest] = asyncio.PriorityQueue()

    def priority_for(self, request: GenerationRequest) -> int:
        return self._priorities.get(request.priority.value, max(self._priorities.values()) + 1)

    async def put(self, request: GenerationRequest, *, payload: Any = None) -> QueuedRequest:
        item = QueuedRequest(
            priority=self.priority_for(request),
            sequence=next(self._sequence),
            enqueued_at=self._clock(),
            request=request,
            payload=payload,
        )
        await self._queue.put(item)
        return item

    async def get(self) -> QueuedRequest:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()

    async def join(self) -> None:
        await self._queue.join()

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()
