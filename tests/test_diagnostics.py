from tf.diagnostics import ForecastDiagnostics


def test_track_lifecycle_and_latency_summary():
    stats = ForecastDiagnostics()
    stats.record({1, 2}, 0.003, 2)
    stats.record({2}, 0.001, 1)
    stats.record({1, 2, 3}, 0.002, 2)
    assert stats.frames == 3
    assert stats.forecast_calls == 5
    assert stats.new_ids == 3
    assert stats.lost_ids == 1
    assert stats.id_reappearances == 1
    assert "2.000 ms/frame" in stats.summary()


def test_empty_diagnostics():
    stats = ForecastDiagnostics()
    assert "0 frames" in stats.summary()
