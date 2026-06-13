import asyncio

from gateway.backends.registry import ReplicaPool
from gateway.cache.prefix_tracker import PrefixTracker
from gateway.config import load_config
from gateway.routing.classifier import HeuristicDifficultyClassifier
from gateway.routing.difficulty import DifficultyRouter
from gateway.routing.prefix_aware import PrefixAwareRouter
from gateway.schemas import ChatCompletionRequest


def make_request(content: str, *, max_tokens: int = 64, request_id: str = "req_route"):
    return ChatCompletionRequest.model_validate(
        {"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens}
    ).to_generation_request(request_id=request_id)


def test_heuristic_classifier_scores_hard_request_above_easy_request() -> None:
    classifier = HeuristicDifficultyClassifier()
    easy = make_request("say hello", max_tokens=16)
    hard = make_request(
        "Analyze this Python architecture and debug the async API tradeoff. " * 80,
        max_tokens=1024,
    )

    assert classifier.score(easy) < classifier.score(hard)
    assert classifier.score(easy) < 0.6
    assert classifier.score(hard) >= 0.6


def test_difficulty_router_maps_easy_and_hard_requests_to_tiers() -> None:
    pool = ReplicaPool.from_config(load_config())
    router = DifficultyRouter(threshold=0.6)

    easy = make_request("summarize this sentence", max_tokens=32)
    hard = make_request(
        "Reason through this code refactor and explain the architecture tradeoffs. " * 80,
        max_tokens=1024,
    )

    assert router.select_tier(easy, pool) == "small"
    assert router.select_tier(hard, pool) == "large"
    assert router.select(easy, pool).tier == "small"
    assert router.select(hard, pool).tier == "large"


def test_prefix_tracker_expires_entries() -> None:
    now = 100.0

    def clock() -> float:
        return now

    tracker = PrefixTracker(ttl_seconds=10, max_entries=10, clock=clock)
    pool = ReplicaPool.from_config(load_config())
    request = make_request("shared system prompt")

    tracker.update(request, pool.get("mock-small-0"))
    assert tracker.lookup(request).replica_id == "mock-small-0"

    now = 111.0

    assert tracker.lookup(request) is None
    assert len(tracker) == 0


def test_prefix_tracker_evicts_lru_entries() -> None:
    tracker = PrefixTracker(ttl_seconds=60, max_entries=1)
    pool = ReplicaPool.from_config(load_config())

    first = make_request("first prefix")
    second = make_request("second prefix")

    tracker.update(first, pool.get("mock-small-0"))
    tracker.update(second, pool.get("mock-small-1"))

    assert tracker.lookup(first) is None
    assert tracker.lookup(second).replica_id == "mock-small-1"


def test_prefix_aware_router_prefers_cached_replica() -> None:
    pool = ReplicaPool.from_config(load_config())
    tracker = PrefixTracker()
    router = PrefixAwareRouter(tracker=tracker)
    request = make_request("shared prefix conversation")

    tracker.update(request, pool.get("mock-small-1"))

    assert router.select(request, pool, tier="small").replica_id == "mock-small-1"


def test_prefix_aware_router_falls_back_when_cached_replica_is_unhealthy() -> None:
    pool = ReplicaPool.from_config(load_config())
    tracker = PrefixTracker()
    router = PrefixAwareRouter(tracker=tracker)
    request = make_request("shared prefix conversation")

    tracker.update(request, pool.get("mock-small-1"))
    pool.mark_health("mock-small-1", False)

    assert router.select(request, pool, tier="small").replica_id == "mock-small-0"


async def consume_one_request(pool: ReplicaPool, replica_id: str) -> None:
    replica = pool.get(replica_id)
    assert replica.reserve() is True
    try:
        async for _ in replica.generate(make_request("busy replica", max_tokens=1)):
            pass
    finally:
        replica.release()


def test_prefix_aware_router_falls_back_when_affinity_would_hotspot() -> None:
    async def run() -> None:
        pool = ReplicaPool.from_config(load_config())
        tracker = PrefixTracker()
        router = PrefixAwareRouter(tracker=tracker)
        request = make_request("shared prefix conversation")

        tracker.update(request, pool.get("mock-small-1"))
        task = asyncio.create_task(consume_one_request(pool, "mock-small-1"))
        await asyncio.sleep(0)

        assert pool.get("mock-small-1").in_flight == 1
        assert router.select(request, pool, tier="small").replica_id == "mock-small-0"

        await task

    asyncio.run(run())
