import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from .config import ForecastConfig
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
):
    """Track objects in a video and forecast where they will move next.

    Each object is tracked with YOLO, smoothed with a constant-velocity Kalman
    filter, and its future path is predicted by rolling that filter forward.
    Boxes, past tracks and forecasts are drawn on each frame, then shown and/or
    saved to a video.

    Args:
        model_path (str): YOLO model file or Ultralytics model name.
        source (str): Path or URL to the input video (URLs are downloaded).
        output_path (str): Where to save the annotated output video.
        config (ForecastConfig | None): Settings; defaults are used if None.
        show (bool): Display frames in a window.
        save (bool): Write annotated frames to ``output_path``.

    Example:
        >>> from tf.inference import run_inference
        >>> run_inference(model_path="yolo26n.pt", source="https://tinyurl.com/bddswzba")
    """
    config = config or ForecastConfig()

    model = YOLO(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

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
        config.history, fps, config.process_noise, config.measurement_noise
    )
    draw = show or save

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            conf=config.conf,
            classes=config.classes,
            tracker=config.tracker,
        )[0]

        active_ids = set()
        ann = Annotator(frame)

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy().astype(int)
            clss = results.boxes.cls.cpu().numpy().astype(int)

            for bbox, tid, cls in zip(boxes, ids, clss):
                x1, y1, x2, y2 = map(int, bbox)
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

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
                        fpts = clamp_points(
                            kf.forecast(config.forecast_steps), width, height
                        )
                        draw_forecast(
                            frame,
                            fpts,
                            config.forecast_color,
                            config.forecast_thickness,
                            config.forecast_radius,
                        )

        tracker_manager.cleanup(active_ids)

        if save:
            writer.write(frame)
        if show:
            cv2.imshow("Tracking + Forecast", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if save:
        writer.release()
    if show:
        cv2.destroyAllWindows()
