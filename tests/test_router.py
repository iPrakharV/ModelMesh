from gateway.app.router import RoundRobinRouter


def test_round_robin_returns_workers_in_order() -> None:
    router = RoundRobinRouter(["http://a", "http://b"])

    assert [worker.url for worker in router.candidates(2)] == ["http://a", "http://b"]
    assert [worker.url for worker in router.candidates(2)] == ["http://a", "http://b"]


def test_round_robin_requires_worker() -> None:
    try:
        RoundRobinRouter([])
    except ValueError as exc:
        assert "worker URL" in str(exc)
    else:
        raise AssertionError("expected ValueError")
