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
    source: str = "https://tinyurl.com/2f3yrppv",
    output_path: str = "forecast-results.mp4",
    config: ForecastConfig | None = None,
    show: bool = True,
    save: bool = True,
):
    """Run object tracking and trajectory forecasting on a video source. This function performs real-time object detection 
    and tracking using a YOLO model and forecasts future object trajectories based on historical tracking data. The pipeline 
    includes velocity estimation, trajectory smoothing, and forward prediction of object motion.

    The processed frames are displayed in a window and saved to an output video.

    Workflow:
        1. Load YOLO model and move it to the available device (CPU or GPU).
        2. Open the input video source or download it if it is a URL.
        3. Perform object detection and tracking using the selected tracker.
        4. Maintain tracking history for each object.
        5. Estimate velocity using recent trajectory points.
        6. Forecast future positions when enough tracking history is available.
        7. Render bounding boxes, trajectory history, and forecast points.
        8. Save and display processed frames.

    Args:
        model_path (str):
            Path to the YOLO model file used for detection and tracking.
            This can be a local file or a supported Ultralytics model name.

        source (str):
            Path or URL to the input video. If a URL is provided, the video
            will be downloaded automatically before processing.

        output_path (str):
            Path to save the output video with tracking and forecasting results.

        config (ForecastConfig):
            Configuration object containing parameters for detection,
            tracking, trajectory history, velocity estimation, and
            forecasting behavior.

        show (bool):
            Enable output results display.

        save (bool):
            Enable output results writing in the video file.

    Example:
        >>> from tf.inference import run_inference
        >>> run_inference(
        ...     model_path="yolo26n.pt",
        ...     source="https://tinyurl.com/bddswzba",
        ...     output_path="results.mp4"
        ... )
    """
    
    model = YOLO(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    source = download_if_url(source)  # Download if the source is a URL i.e from GitHub assets

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    
    tracker_manager = TrackManager(config.history, config.ema_alpha)

    while True:

        ret, frame = cap.read()
        
        if not ret:
            break
        
        results = model.track(frame, persist=True, conf=config.conf, classes=config.classes, tracker=config.tracker)[0]

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

                if show:
                    bbox_color = colors(cls, True)
                    label = f"#{tid}"
    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)  # Draw bbox
    
                    tw, th = cv2.getTextSize(label, 0, config.font_scale, config.font_thickness)[0]
    
                    rect_w, rect_h = tw + 2 * config.padding, th + 2 * config.padding
    
                    cv2.rectangle(frame, (x1, y1), (x1 + rect_w, y1 + rect_h), bbox_color, -1)
    
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

                    past_pts = clamp_points(list(tracker_manager.history[tid]), width, height)
                    draw_polyline(frame, past_pts, bbox_color)

                if len(past_pts) >= config.min_points:
                    vx, vy = estimate_velocity(past_pts, fps, config.vel_window)

                    if np.hypot(vx, vy) > 1.0:
                        fpts = forecast_points(past_pts[-1], vx, vy, fps, config.forecast_steps)
                        fpts = clamp_points(fpts, width, height)
                        if show:
                            draw_forecast(frame, fpts, config.forecast_color)

        tracker_manager.cleanup(active_ids)

        if save:
            writer.write(frame)
        if show:
            cv2.imshow("Tracking + Forecast", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
