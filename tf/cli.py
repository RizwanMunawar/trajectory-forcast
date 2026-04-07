import argparse
from .inference import run_inference
from .config import ForecastConfig
import logging

logger = logging.getLogger(name="trajectory-forecast")


def main():
    parser = argparse.ArgumentParser(description="Trajectory Forecast Package")

    parser.add_argument("--model", help="All models supported by Ultralytics Package.")
    parser.add_argument("--source", default="https://tinyurl.com/bddswzba", required=False)
    parser.add_argument("--output", default="output.mp4")
    parser.add_argument("--config", type=str, help="Path to YAML config")
    parser.add_argument("--show", type=bool, default=True, help="Display the results")
    parser.add_argument("--save", type=bool, default=True, help="Display the results")
    

    # Optional overrides
    parser.add_argument("--conf", type=float)
    parser.add_argument("--history", type=int)
    parser.add_argument("--forecast_steps", type=int)
    parser.add_argument("--ema_alpha", type=float)

    args = parser.parse_args()

    #Warning if model not provided.
    if not args.model:
        logger.warning("No model selected, using YOLO26s.pt...")

    # Load config
    if args.config:
        config = ForecastConfig.from_yaml(args.config)
    else:
        config = ForecastConfig()

    # CLI overrides YAML
    if args.conf is not None:
        config.conf = args.conf
    if args.history is not None:
        config.history = args.history
    if args.forecast_steps is not None:
        config.forecast_steps = args.forecast_steps
    if args.ema_alpha is not None:
        config.ema_alpha = args.ema_alpha

    run_inference(
        model_path=args.model,
        source=args.source,
        output_path=args.output,
        config=config,
        show=args.show,
        save=args.save,
    )
