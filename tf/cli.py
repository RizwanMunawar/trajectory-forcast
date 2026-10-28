import argparse
import logging

from .config import ForecastConfig
from .inference import run_inference

logger = logging.getLogger(name="trajectory-forecast")

DEFAULT_MODEL = "yolo26n.pt"


def main():
    """Parse command-line arguments and run tracking with forecasting."""
    parser = argparse.ArgumentParser(description="Trajectory Forecast Package")

    parser.add_argument("--model", help="Any model supported by Ultralytics.")
    parser.add_argument("--source", default="https://tinyurl.com/bddswzba")
    parser.add_argument("--output", default="output.mp4")
    parser.add_argument("--config", type=str, help="Path to YAML config.")
    parser.add_argument("--show", action="store_true", help="Display the results.")
    parser.add_argument("--save", action="store_true", help="Save the results.")
    parser.add_argument("--profile", action="store_true", help="Summarize forecast timing and track-ID lifecycle.")

    # Optional overrides that take priority over the YAML config.
    parser.add_argument("--conf", type=float)
    parser.add_argument("--history", type=int)
    parser.add_argument("--forecast_steps", type=int)
    parser.add_argument("--process_noise", type=float)
    parser.add_argument("--measurement_noise", type=float)
    parser.add_argument("--imgsz", type=int, help="Inference image size; smaller sizes can improve speed but may reduce accuracy.")
    parser.add_argument("--half", action="store_true", help="Use half precision where supported.")
    parser.add_argument("--device", type=str, help="Inference device, e.g. cpu, 0, or mps. Default: automatic selection.")

    args = parser.parse_args()

    model = args.model
    if not model:
        logger.warning("No model selected, using %s ...", DEFAULT_MODEL)
        model = DEFAULT_MODEL

    config = ForecastConfig.from_yaml(args.config) if args.config else ForecastConfig()

    # CLI overrides win over the YAML config.
    if args.conf is not None:
        config.conf = args.conf
    if args.history is not None:
        config.history = args.history
    if args.forecast_steps is not None:
        config.forecast_steps = args.forecast_steps
    if args.process_noise is not None:
        config.process_noise = args.process_noise
    if args.measurement_noise is not None:
        config.measurement_noise = args.measurement_noise
    if args.imgsz is not None:
        if args.imgsz <= 0:
            parser.error("--imgsz must be a positive integer")
        config.imgsz = args.imgsz
    if args.half:
        config.half = True
    if args.device is not None:
        config.device = args.device

    run_inference(
        model_path=model,
        source=args.source,
        output_path=args.output,
        config=config,
        show=args.show,
        save=args.save,
        profile=args.profile,
    )
