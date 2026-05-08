from gateway.app.metrics import MetricsStore, percentile


def test_percentile_handles_empty_and_ordered_values() -> None:
    assert percentile([], 0.95) == 0
    assert percentile([30, 10, 20], 0.50) == 20


def test_metrics_snapshot_tracks_cache_and_latency() -> None:
    store = MetricsStore()

    store.record_request(10, cached=False)
    store.record_request(30, cached=True)

    snapshot = store.snapshot()
    assert snapshot["requests"] == 2
    assert snapshot["cache_hits"] == 1
    assert snapshot["cache_hit_rate"] == 0.5
    assert snapshot["p50_latency_ms"] == 10
