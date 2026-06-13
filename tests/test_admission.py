import pytest

from gateway.scheduling.admission import AdmissionController, AdmissionDecision


def test_admission_accepts_when_capacity_is_available() -> None:
    controller = AdmissionController(max_queue_depth=2, target_p99_ms=100)

    result = controller.decide(queue_depth=0, has_capacity=True)

    assert result.decision is AdmissionDecision.ACCEPT
    assert result.reason == "capacity_available"


def test_admission_queues_when_capacity_is_unavailable_but_queue_has_room() -> None:
    controller = AdmissionController(max_queue_depth=2, target_p99_ms=100)

    result = controller.decide(queue_depth=1, has_capacity=False, estimated_wait_ms=50)

    assert result.decision is AdmissionDecision.QUEUE
    assert result.reason == "queued_for_capacity"


def test_admission_sheds_when_queue_is_full() -> None:
    controller = AdmissionController(max_queue_depth=2, target_p99_ms=100)

    result = controller.decide(queue_depth=2, has_capacity=False)

    assert result.decision is AdmissionDecision.SHED
    assert result.reason == "queue_full"


@pytest.mark.parametrize("estimated_wait_ms", [101, 250])
def test_admission_sheds_when_estimated_wait_violates_slo(estimated_wait_ms: int) -> None:
    controller = AdmissionController(max_queue_depth=10, target_p99_ms=100)

    result = controller.decide(
        queue_depth=1,
        has_capacity=False,
        estimated_wait_ms=estimated_wait_ms,
    )

    assert result.decision is AdmissionDecision.SHED
    assert result.reason == "slo_risk"
