import pytest

from gateway.backends.registry import NoHealthyReplicaError, ReplicaPool
from gateway.config import load_config
from gateway.schemas import ChatCompletionRequest


def make_request(max_tokens: int = 1):
    return ChatCompletionRequest.model_validate(
        {"messages": [{"role": "user", "content": "route this"}], "max_tokens": max_tokens}
    ).to_generation_request(request_id="req_pool")


def test_replica_pool_builds_mock_replicas_from_config() -> None:
    pool = ReplicaPool.from_config(load_config())

    assert len(pool.replicas) == 3
    assert pool.tiers == {"small", "large"}
    assert [replica.replica_id for replica in pool.replicas_for_tier("small")] == [
        "mock-small-0",
        "mock-small-1",
    ]
    assert [replica.replica_id for replica in pool.replicas_for_tier("large")] == [
        "mock-large-0"
    ]


def test_replica_pool_filters_unhealthy_replicas() -> None:
    pool = ReplicaPool.from_config(load_config())

    pool.mark_health("mock-small-0", False)

    assert [replica.replica_id for replica in pool.healthy_replicas("small")] == [
        "mock-small-1"
    ]
    assert pool.least_loaded("small").replica_id == "mock-small-1"


def test_replica_pool_raises_when_no_replica_is_available() -> None:
    pool = ReplicaPool.from_config(load_config())

    pool.mark_health("mock-large-0", False)

    with pytest.raises(NoHealthyReplicaError, match="large"):
        pool.least_loaded("large")


@pytest.mark.asyncio
async def test_replica_pool_least_loaded_avoids_busy_replica() -> None:
    pool = ReplicaPool.from_config(load_config())
    busy_replica = pool.get("mock-small-0")

    assert busy_replica.reserve() is True
    assert busy_replica.in_flight == 1
    assert pool.least_loaded("small").replica_id == "mock-small-1"
    busy_replica.release()


@pytest.mark.asyncio
async def test_replica_pool_refresh_health_reads_backend_health() -> None:
    pool = ReplicaPool.from_config(load_config())
    replica = pool.get("mock-small-0")
    replica.set_healthy(False)

    health = await pool.refresh_health()

    assert health["mock-small-0"] is False
    assert [replica.replica_id for replica in pool.healthy_replicas("small")] == [
        "mock-small-1"
    ]
