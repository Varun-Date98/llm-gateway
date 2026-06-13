from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from gateway.backends.base import Backend
from gateway.backends.registry import NoHealthyReplicaError, ReplicaPool
from gateway.routing.base import Router
from gateway.scheduling.admission import (
    AdmissionController,
    AdmissionDecision,
    AdmissionResult,
)
from gateway.scheduling.queue import PriorityRequestQueue, QueuedRequest
from gateway.schemas import GenerationRequest, Token


class SchedulerClosed(RuntimeError):
    """Raised when submitting to a stopped scheduler."""


class AdmissionRejected(RuntimeError):
    """Raised when admission control sheds a request."""

    def __init__(self, result: AdmissionResult) -> None:
        super().__init__(result.reason)
        self.result = result


@dataclass(frozen=True)
class _EndOfStream:
    pass


END_OF_STREAM = _EndOfStream()


@dataclass
class ScheduledStream:
    request: GenerationRequest
    admission: AdmissionResult
    enqueued_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    completed_at: float | None = None
    backend_id: str | None = None
    _items: asyncio.Queue[Token | BaseException | _EndOfStream] = field(
        default_factory=asyncio.Queue
    )

    def __aiter__(self) -> AsyncIterator[Token]:
        return self.stream()

    async def stream(self) -> AsyncIterator[Token]:
        while True:
            item = await self._items.get()
            if item is END_OF_STREAM:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    async def send(self, token: Token) -> None:
        await self._items.put(token)

    async def fail(self, error: BaseException) -> None:
        self.completed_at = time.monotonic()
        await self._items.put(error)
        await self._items.put(END_OF_STREAM)

    async def close(self) -> None:
        self.completed_at = time.monotonic()
        await self._items.put(END_OF_STREAM)


class Scheduler:
    """Admission, queueing, routing, and backend dispatch loop."""

    def __init__(
        self,
        *,
        pool: ReplicaPool,
        router: Router,
        admission: AdmissionController,
        queue: PriorityRequestQueue | None = None,
        dispatch_interval_ms: int = 5,
    ) -> None:
        self.pool = pool
        self.router = router
        self.admission = admission
        self.queue = queue or PriorityRequestQueue()
        self.dispatch_interval = dispatch_interval_ms / 1000.0
        self._running = False
        self._dispatch_task: asyncio.Task[None] | None = None
        self._active_tasks: set[asyncio.Task[None]] = set()

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())

    async def stop(self, *, drain: bool = True) -> None:
        self._running = False
        if drain:
            await self.queue.join()
            if self._active_tasks:
                await asyncio.gather(*self._active_tasks, return_exceptions=True)
        if self._dispatch_task is not None:
            self._dispatch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._dispatch_task
            self._dispatch_task = None

    async def submit(self, request: GenerationRequest) -> ScheduledStream:
        if not self._running:
            raise SchedulerClosed("scheduler is not running")

        result = self.admission.decide(
            queue_depth=self.queue.qsize(),
            has_capacity=self._has_capacity(request),
            estimated_wait_ms=self._estimate_wait_ms(),
        )
        
        if result.decision is AdmissionDecision.SHED:
            raise AdmissionRejected(result)

        stream = ScheduledStream(request=request, admission=result)
        await self.queue.put(request, payload=stream)
        return stream

    async def _dispatch_loop(self) -> None:
        while self._running:
            item = await self.queue.get()
            try:
                backend = await self._wait_for_backend(item.request)
                stream = self._stream_from_item(item)
                task = asyncio.create_task(self._run_backend_stream(item, stream, backend))
                self._active_tasks.add(task)
                task.add_done_callback(self._active_tasks.discard)
                await asyncio.sleep(0)
            finally:
                self.queue.task_done()

    def _stream_from_item(self, item: QueuedRequest) -> ScheduledStream:
        if not isinstance(item.payload, ScheduledStream):
            raise TypeError("queued scheduler item is missing its stream handle")
        return item.payload

    async def _wait_for_backend(self, request: GenerationRequest) -> Backend:
        while self._running:
            try:
                return self.router.select(request, self.pool)
            except NoHealthyReplicaError:
                await asyncio.sleep(self.dispatch_interval)
        raise SchedulerClosed("scheduler stopped before a backend became available")

    async def _run_backend_stream(
        self,
        item: QueuedRequest,
        stream: ScheduledStream,
        backend: Backend,
    ) -> None:
        stream.started_at = time.monotonic()
        stream.backend_id = backend.replica_id
        try:
            async for token in backend.generate(item.request):
                await stream.send(token)
            self._record_routing_affinity(item.request, backend)
            await stream.close()
        except BaseException as error:
            await stream.fail(error)

    def _record_routing_affinity(self, request: GenerationRequest, backend: Backend) -> None:
        recorder = getattr(self.router, "record", None)
        if callable(recorder):
            recorder(request, backend)

    def _has_capacity(self, request: GenerationRequest) -> bool:
        try:
            self.router.select(request, self.pool)
        except NoHealthyReplicaError:
            return False
        return True

    def _estimate_wait_ms(self) -> float:
        if self.queue.qsize() == 0:
            return 0.0
        return self.queue.qsize() * self.dispatch_interval * 1000.0
