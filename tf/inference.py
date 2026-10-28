import time

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from .config import ForecastConfig
from .diagnostics import ForecastDiagnostics
from .drawing import clamp_points, draw_forecast, draw_polyline
from .tracker import TrackManager
from .utils import download_if_url


def run_inference(
    model_path: str = "yolo26n.pt",
    source: str = "https://tinyurl.com/2f3yrppv",
    output_path: str = "forecast-results.mp4",
    config: ForecastConfig | None = None,
    show: bool = True,
    save: bool = True,
    profile: bool = False,
):
    """Track objects in a video and forecast where they will move next.

    Each object is tracked with YOLO, smoothed with an acceleration-aware Kalman
    filter, and its future path is predicted from the filtered motion state.
    Boxes, past tracks and forecasts are drawn on each frame, then shown and/or
    saved to a video.

    Args:
        model_path (str): YOLO model file or Ultralytics model name.
        source (str): Path or URL to the input video (URLs are downloaded).
        output_path (str): Where to save the annotated output video.
        config (ForecastConfig | None): Settings; defaults are used if None.
        show (bool): Display frames in a window.
        save (bool): Write annotated frames to ``output_path``.
        profile (bool): Print aggregate forecasting and track-ID diagnostics.

    Example:
        >>> from tf.inference import run_inference
        >>> run_inference(model_path="yolo26n.pt", source="https://tinyurl.com/bddswzba")
    """
    config = config or ForecastConfig()

    model = YOLO(model_path)
    source = download_if_url(source)  # Download first if the source is a URL.

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Fill any unset drawing sizes based on the video resolution.
    config.resolve_visuals(width, height)

    if save:
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

    tracker_manager = TrackManager(
        config.history,
        fps,
        config.process_noise,
        config.measurement_noise,
        config.max_gap_frames,
    )
    draw = show or save
    diagnostics = ForecastDiagnostics() if profile else None
    stage_seconds = {name: 0.0 for name in ("decode", "inference_tracking", "postprocess_forecast_drawing", "write", "display", "total")}
    track_kwargs = {
        "persist": True,
        "conf": config.conf,
        "classes": config.classes,
        "tracker": config.tracker,
    }
    if config.imgsz is not None:
        track_kwargs["imgsz"] = config.imgsz
    if config.half:
        track_kwargs["half"] = True
    if config.device is not None:
        track_kwargs["device"] = config.device

    while True:
        frame_started = time.perf_counter() if profile else 0.0
        stage_started = time.perf_counter() if profile else 0.0
        ret, frame = cap.read()
        if profile:
            stage_seconds["decode"] += time.perf_counter() - stage_started
        if not ret:
            break

        stage_started = time.perf_counter() if profile else 0.0
        results = model.track(frame, **track_kwargs)[0]
        if profile:
            stage_seconds["inference_tracking"] += time.perf_counter() - stage_started
        processing_started = time.perf_counter() if profile else 0.0

        active_ids = set()
        ann = Annotator(frame) if draw else None
        forecast_time = 0.0
        forecast_calls = 0

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy().astype(int)
            clss = results.boxes.cls.cpu().numpy().astype(int)

            for bbox, tid, cls in zip(boxes, ids, clss):
                # Preserve subpixel detector coordinates for motion estimation.
                # Integer conversion is only needed for drawing the bounding box.
                x1f, y1f, x2f, y2f = bbox
                cx, cy = (x1f + x2f) / 2.0, (y1f + y2f) / 2.0
                x1, y1, x2, y2 = map(int, bbox)

                active_ids.add(tid)
                kf = tracker_manager.update(tid, cx, cy)

                if not draw:
                    continue

                bbox_color = colors(cls, True)
                label = f"#{tid}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, config.line_thickness)

                tw, th = cv2.getTextSize(
                    label, 0, config.font_scale, config.font_thickness
                )[0]
                rect_w, rect_h = tw + 2 * config.padding, th + 2 * config.padding
                cv2.rectangle(
                    frame, (x1, y1), (x1 + rect_w, y1 + rect_h), bbox_color, -1
                )
                text_x, text_y = x1 + (rect_w - tw) // 2, y1 + (rect_h + th) // 2
                cv2.putText(
                    frame,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    config.font_scale,
                    ann.get_txt_color(bbox_color),
                    config.font_thickness,
                    cv2.LINE_AA,
                )

                past_pts = clamp_points(
                    list(tracker_manager.history[tid]), width, height
                )
                draw_polyline(frame, past_pts, bbox_color, config.line_thickness)

                if len(tracker_manager.history[tid]) >= config.min_points:
                    vx, vy = kf.velocity()
                    if np.hypot(vx, vy) > config.min_speed:
                        forecast_start = time.perf_counter()
                        future = kf.forecast(config.forecast_steps)
                        forecast_time += time.perf_counter() - forecast_start
                        forecast_calls += 1
                        fpts = clamp_points(future, width, height)
                        draw_forecast(
                            frame,
                            fpts,
                            config.forecast_color,
                            config.forecast_thickness,
                            config.forecast_radius,
                        )

        tracker_manager.cleanup(active_ids)
        if profile:
            stage_seconds["postprocess_forecast_drawing"] += time.perf_counter() - processing_started
        if diagnostics is not None:
            diagnostics.record(active_ids, forecast_time, forecast_calls)

        # Per-frame stdout can become a bottleneck on long/high-FPS videos.
        # Keep detailed timing output opt-in with profiling enabled.
        if diagnostics is not None:
            print(f"Trajectory forecast: {forecast_time * 1000:.2f}ms per frame")

        if save:
            stage_started = time.perf_counter() if profile else 0.0
            writer.write(frame)
            if profile:
                stage_seconds["write"] += time.perf_counter() - stage_started
        if show:
            stage_started = time.perf_counter() if profile else 0.0
            cv2.imshow("Tracking + Forecast", frame)
            should_quit = cv2.waitKey(1) & 0xFF == ord("q")
            if profile:
                stage_seconds["display"] += time.perf_counter() - stage_started
        else:
            should_quit = False
        if profile:
            stage_seconds["total"] += time.perf_counter() - frame_started
        if should_quit:
            break

    if diagnostics is not None:
        print(diagnostics.summary())
        frames = max(diagnostics.frames, 1)
        print("Pipeline profiling (average ms/frame): " + ", ".join(
            f"{name}={1000.0 * seconds / frames:.2f}"
            for name, seconds in stage_seconds.items()
        ))

    cap.release()
    if save:
        writer.release()
    if show:
        cv2.destroyAllWindows()
