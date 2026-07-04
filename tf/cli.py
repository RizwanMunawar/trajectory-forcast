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

    # Optional overrides that take priority over the YAML config.
    parser.add_argument("--conf", type=float)
    parser.add_argument("--history", type=int)
    parser.add_argument("--forecast_steps", type=int)
    parser.add_argument("--process_noise", type=float)
    parser.add_argument("--measurement_noise", type=float)

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

    run_inference(
        model_path=model,
        source=args.source,
        output_path=args.output,
        config=config,
        show=args.show,
        save=args.save,
    )
