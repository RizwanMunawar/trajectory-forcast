"""Low-overhead per-frame forecasting diagnostics.

Tracks the forecast-only latency and track lifecycle without changing the
Kalman filter, tracker associations, or drawing decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ForecastDiagnostics:
    frames: int = 0
    forecast_seconds: float = 0.0
    forecast_calls: int = 0
    new_ids: int = 0
    lost_ids: int = 0
    id_reappearances: int = 0
    _previous_ids: set[int] = field(default_factory=set, repr=False)
    _seen_ids: set[int] = field(default_factory=set, repr=False)

    def record(self, ids: set[int], duration: float, calls: int) -> None:
        self.frames += 1
        self.forecast_seconds += duration
        self.forecast_calls += calls
        entered = ids - self._previous_ids
        self.new_ids += len(entered - self._seen_ids)
        self.id_reappearances += len(entered & self._seen_ids)
        self.lost_ids += len(self._previous_ids - ids)
        self._seen_ids.update(ids)
        self._previous_ids = set(ids)

    def summary(self) -> str:
        average = 1000.0 * self.forecast_seconds / max(self.frames, 1)
        return (
            f"Forecast profiling: {self.frames} frames, {self.forecast_calls} predictions, "
            f"average {average:.3f} ms/frame, "
            f"{self.new_ids} new IDs, {self.lost_ids} disappearances, "
            f"{self.id_reappearances} ID reappearances"
        )
