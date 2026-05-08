from gateway.app.router import WorkerPool


def test_worker_pool_prefers_lower_latency_and_no_failures() -> None:
    router = WorkerPool(["http://a", "http://b"])
    first, second = router.workers

    router.mark_success(first, 80)
    router.mark_success(second, 20)

    assert router.candidates(1)[0].url == "http://b"


def test_worker_pool_penalizes_failures() -> None:
    router = WorkerPool(["http://a", "http://b"])
    first, second = router.workers

    router.mark_failure(first, "timeout")
    router.mark_success(second, 50)

    assert router.candidates(1)[0].url == "http://b"
    assert router.snapshot()[0]["failures"] == 1


def test_worker_pool_requires_worker() -> None:
    try:
        WorkerPool([])
    except ValueError as exc:
        assert "worker URL" in str(exc)
    else:
        raise AssertionError("expected ValueError")
