"""Controlled straight-motion check; run with python -m benchmarks.synthetic_forecast."""

import math
import random

from tf.tracker import TrackManager


def evaluate(max_gap_frames, missing):
    rng = random.Random(123)
    fps = 20.0
    manager = TrackManager(30, fps, 1.0, 10.0, max_gap_frames=max_gap_frames)
    errors = []
    available = 0

    for frame in range(200):
        active = set()
        if frame not in missing:
            active.add(7)
            x = 100 + 60 * frame / fps + rng.gauss(0, 1.5)
            y = 100 + 25 * frame / fps + rng.gauss(0, 1.5)
            kf = manager.update(7, x, y)
            if frame >= 8 and len(manager.history[7]) >= 5:
                available += 1
                for step, (px, py) in enumerate(kf.forecast(10), 1):
                    tx = 100 + 60 * (frame + step) / fps
                    ty = 100 + 25 * (frame + step) / fps
                    errors.append(math.hypot(px - tx, py - ty))
        manager.cleanup(active)

    return available, sum(errors) / len(errors)


if __name__ == "__main__":
    gaps = {n + d for n in range(25, 180, 25) for d in (0, 1)}
    for label, max_gap, missing in (
        ("continuous", 0, set()),
        ("immediate cleanup", 0, gaps),
        ("retain five frames", 5, gaps),
    ):
        forecasts, error = evaluate(max_gap, missing)
        print(f"{label}: {forecasts} forecasts, {error:.3f} px mean 10-step error")
