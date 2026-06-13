import pytest

from gateway.backends.registry import ReplicaPool
from gateway.cache.prefix_tracker import PrefixTracker
from gateway.config import load_config
from gateway.routing.difficulty import DifficultyRouter
from gateway.routing.prefix_aware import PrefixAwareRouter
from gateway.scheduling.admission import AdmissionController, AdmissionDecision
from gateway.scheduling.queue import PriorityRequestQueue
from gateway.scheduling.scheduler import AdmissionRejected, Scheduler, SchedulerClosed
from gateway.schemas import ChatCompletionRequest


def make_request(
    content: str = "hello",
    *,
    max_tokens: int = 2,
    priority: str = "interactive",
    request_id: str = "req_scheduler",
):
    return ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "priority": priority,
        }
    ).to_generation_request(request_id=request_id)


@pytest.mark.asyncio
async def test_priority_queue_dispatches_interactive_before_batch() -> None:
    queue = PriorityRequestQueue({"interactive": 0, "batch": 1})
    batch = make_request(priority="batch", request_id="batch")
    interactive = make_request(priority="interactive", request_id="interactive")

    await queue.put(batch)
    await queue.put(interactive)

    first = await queue.get()
    second = await queue.get()

    assert first.request.request_id == "interactive"
    assert second.request.request_id == "batch"


@pytest.mark.asyncio
async def test_scheduler_streams_tokens_and_records_prefix_affinity() -> None:
    pool = ReplicaPool.from_config(load_config())
    tracker = PrefixTracker()
    router = DifficultyRouter(
        threshold=0.6,
        within_tier_router=PrefixAwareRouter(tracker=tracker),
    )
    scheduler = Scheduler(
        pool=pool,
        router=router,
        admission=AdmissionController(max_queue_depth=10, target_p99_ms=1000),
    )
    request = make_request("shared prefix", max_tokens=2)

    scheduler.start()
    stream = await scheduler.submit(request)
    tokens = [token async for token in stream]
    await scheduler.stop()

    assert stream.admission.decision is AdmissionDecision.ACCEPT
    assert [token.index for token in tokens] == [0, 1]
    assert stream.backend_id is not None
    assert tracker.lookup(request).replica_id == stream.backend_id
    assert pool.get(stream.backend_id).in_flight == 0


@pytest.mark.asyncio
async def test_scheduler_sheds_when_queue_is_full_and_no_replica_has_capacity() -> None:
    pool = ReplicaPool.from_config(load_config())
    pool.mark_health("mock-small-0", False)
    pool.mark_health("mock-small-1", False)
    scheduler = Scheduler(
        pool=pool,
        router=DifficultyRouter(threshold=0.6),
        admission=AdmissionController(max_queue_depth=0, target_p99_ms=1000),
    )

    scheduler.start()
    with pytest.raises(AdmissionRejected) as error:
        await scheduler.submit(make_request("easy request"))
    await scheduler.stop()

    assert error.value.result.decision is AdmissionDecision.SHED
    assert error.value.result.reason == "queue_full"


@pytest.mark.asyncio
async def test_scheduler_rejects_submit_before_start() -> None:
    scheduler = Scheduler(
        pool=ReplicaPool.from_config(load_config()),
        router=DifficultyRouter(threshold=0.6),
        admission=AdmissionController(max_queue_depth=10, target_p99_ms=1000),
    )

    with pytest.raises(SchedulerClosed):
        await scheduler.submit(make_request())
