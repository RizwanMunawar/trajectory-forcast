import cv2
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors, Annotator
from .utils import download_if_url
from .config import ForecastConfig
from .drawing import draw_polyline, draw_forecast, clamp_points
from .forecasting import estimate_velocity, forecast_points
from .tracker import TrackManager


def run_inference(
    model_path: str = "yolo26n.pt",
    source: str = "https://github.com/RizwanMunawar/trajectory-forcast/releases/download/0.0.1/cars-on-highway.mp4",
    output_path: str = "forecast-results.mp4",
    config: ForecastConfig = ForecastConfig(),
):
    print(model_path)
    model = YOLO(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Download if the source is a URL i.e from GitHub assets
    source = download_if_url(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    tracker_manager = TrackManager(config.history, config.ema_alpha)

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
                tracker_manager.update(tid, cx, cy)

                bbox_color = colors(cls, True)
                label = f"{tid}"

                # Draw bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2, cv2.LINE_AA)

                (tw, th), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    config.font_scale,
                    config.font_thickness,
                )

                rect_w = tw + 2 * config.padding
                rect_h = th + 2 * config.padding

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x1 + rect_w, y1 + rect_h),
                    bbox_color,
                    -1,
                )

                text_x = x1 + (rect_w - tw) // 2
                text_y = y1 + (rect_h + th) // 2

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
                    list(tracker_manager.history[tid]),
                    width, height
                )
                draw_polyline(frame, past_pts, bbox_color)

                if len(past_pts) >= config.min_points:
                    vx, vy = estimate_velocity(past_pts, fps, config.vel_window)

                    if np.hypot(vx, vy) > 1.0:
                        fpts = forecast_points(
                            past_pts[-1],
                            vx,
                            vy,
                            fps,
                            config.forecast_steps,
                        )

                        fpts = clamp_points(fpts, width, height)
                        draw_forecast(frame, fpts, config.forecast_color)

        tracker_manager.cleanup(active_ids)

        writer.write(frame)
        cv2.imshow("Tracking + Forecast", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()