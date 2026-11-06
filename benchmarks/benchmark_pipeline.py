"""Reproducible microbenchmark for the tracking/forecast/drawing Python pipeline.

This intentionally excludes YOLO inference, video decoding, and tracker association.
It measures the package's per-track Kalman update, forecast generation, and optional
OpenCV drawing costs with deterministic synthetic trajectories. Use a real video and
the CLI's --profile option to measure end-to-end performance.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import numpy as np

from tf.drawing import clamp_points, draw_forecast, draw_polyline
from tf.tracker import TrackManager


def _run_once(
    track_count: int,
    frames: int,
    warmup: int,
    forecast_steps: int,
    history_size: int,
    draw: bool,
    width: int,
    height: int,
) -> dict[str, list[float]]:
    """Run one deterministic scenario and return per-frame stage durations."""
    manager = TrackManager(
        history_size=history_size,
        fps=30.0,
        process_noise=1.0,
        measurement_noise=10.0,
    )
    # Stable, separated paths make the benchmark repeatable and keep most
    # points inside the frame without introducing random-number overhead.
    offsets = np.linspace(0.0, min(width, height) * 0.35, track_count)
    phase = np.linspace(0.0, 2.0 * np.pi, track_count, endpoint=False)
    frame = np.zeros((height, width, 3), dtype=np.uint8) if draw else None
    timings = {"update": [], "forecast": [], "drawing": [], "total": []}

    for frame_index in range(warmup + frames):
        active_ids = set(range(track_count))
        update_started = time.perf_counter()
        filters = []
        for track_id in range(track_count):
            t = frame_index / 30.0
            x = width * 0.5 + offsets[track_id] * np.cos(t + phase[track_id])
            y = height * 0.5 + offsets[track_id] * np.sin(t + phase[track_id])
            filters.append(manager.update(track_id, float(x), float(y)))
        update_elapsed = time.perf_counter() - update_started

        forecast_started = time.perf_counter()
        forecasts = [
            kf.forecast(forecast_steps)
            for kf in filters
            if len(manager.history[track_id]) >= 2
        ]
        forecast_elapsed = time.perf_counter() - forecast_started

        drawing_started = time.perf_counter()
        if draw:
            # Clear between frames to avoid measuring an increasingly complex
            # image or accumulating old trajectories.
            frame.fill(0)
            for track_id, (kf, future) in enumerate(zip(filters, forecasts)):
                past = clamp_points(
                    list(manager.history[track_id]), width, height
                )
                draw_polyline(frame, past, (0, 200, 0), 2)
                draw_forecast(
                    frame, clamp_points(future, width, height), (0, 0, 255), 2, 4
                )
        drawing_elapsed = time.perf_counter() - drawing_started

        manager.cleanup(active_ids)
        if frame_index >= warmup:
            timings["update"].append(update_elapsed)
            timings["forecast"].append(forecast_elapsed)
            timings["drawing"].append(drawing_elapsed)
            timings["total"].append(
                update_elapsed + forecast_elapsed + drawing_elapsed
            )

    return timings


def _summary(samples: list[float]) -> dict[str, float]:
    milliseconds = [sample * 1000.0 for sample in samples]
    median_ms = statistics.median(milliseconds)
    return {
        "median_ms_per_frame": round(median_ms, 4),
        "p95_ms_per_frame": round(float(np.percentile(milliseconds, 95)), 4),
        "median_frames_per_second": round(1000.0 / median_ms, 2)
        if median_ms > 0
        else 0.0,
    }


def benchmark(args: argparse.Namespace) -> dict:
    results = []
    for draw in (False, True):
        mode = "tracking_forecast_drawing" if draw else "tracking_forecast_only"
        for track_count in args.tracks:
            runs = [
                _run_once(
                    track_count=track_count,
                    frames=args.frames,
                    warmup=args.warmup,
                    forecast_steps=args.forecast_steps,
                    history_size=args.history,
                    draw=draw,
                    width=args.width,
                    height=args.height,
                )
                for _ in range(args.runs)
            ]
            stage_results = {}
            for stage in ("update", "forecast", "drawing", "total"):
                combined = [sample for run in runs for sample in run[stage]]
                stage_results[stage] = _summary(combined)
            results.append(
                {
                    "mode": mode,
                    "tracks": track_count,
                    "frames_per_run": args.frames,
                    "runs": args.runs,
                    "forecast_steps": args.forecast_steps,
                    "history": args.history,
                    "stages": stage_results,
                }
            )
    return {
        "benchmark": "synthetic_tracking_forecast_microbenchmark",
        "scope": (
            "Excludes YOLO inference, detector/tracker association, video decode, "
            "and video encoding. Synthetic deterministic trajectories only."
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", nargs="+", type=int, default=[1, 10, 25, 50, 100])
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--forecast-steps", type=int, default=35)
    parser.add_argument("--history", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--json", dest="json_path", help="Also write results to this JSON file")
    args = parser.parse_args()

    if any(value <= 0 for value in args.tracks):
        parser.error("--tracks values must be positive")
    if min(args.frames, args.runs, args.width, args.height) <= 0 or args.warmup < 0:
        parser.error("--frames, --runs, --width, and --height must be positive; --warmup non-negative")
    if args.forecast_steps < 0 or args.history <= 0:
        parser.error("--forecast-steps must be non-negative and --history positive")

    report = benchmark(args)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")


if __name__ == "__main__":
    main()
