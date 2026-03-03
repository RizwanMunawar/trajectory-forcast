# Trajectory Forecast

Trajectory Forecast is a lightweight, modular extension built on top of Ultralytics YOLO that enables real-time multi-object tracking with future motion prediction.

It combines detection, tracking, motion history modeling, and velocity-based forecasting into a unified pipeline that can be used both as a command-line tool and as a Python library. The system is designed for practical computer vision applications such as traffic analytics, surveillance systems, robotics pipelines, and edge AI deployments.

## Features

- Real-time object detection using Ultralytics YOLO
- Multi-object tracking (ByteTrack / BoT-SORT compatible)
- Trajectory history visualization
- Velocity-based future path forecasting

## Installation

### Recommended (Development Mode)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
````

### Standard Installation (Coming soon)

```bash
pip install trajectory-forecast
```

## Command-Line (CLI) Usage

Run tracking and forecasting on a video:

```bash
trajectory-forecast \
  --model yolo26n.pt \
  --source "https://github.com/RizwanMunawar/trajectory-forcast/releases/download/0.0.1/cars-on-highway.mp4" \
  --output result.mp4
```

Configurable arguments

```bash
```bash
trajectory-forecast \
  --model yolo26n.pt \
  --source "https://github.com/RizwanMunawar/trajectory-forcast/releases/download/0.0.1/cars-on-highway.mp4" \
  --output result.mp4 \
  --conf 0.5 \
  --history 30 \
  --forecast_steps 30 \
```

## Python Usage

The package can also be used programmatically:

```python
from trajectory_forecast import run_inference
from trajectory_forecast.config import ForecastConfig

config = ForecastConfig(
    conf=0.5,
    forecast_steps=50,
    ema_alpha=0.7,
)

run_inference(
    model_path="yolo11n.pt",
    source="video.mp4",
    output_path="output.mp4",
    config=config,
)
```

## Forecasting Methodology

The current forecasting implementation is based on:

* Exponential moving average smoothing of object centers
* Median velocity estimation over a sliding window
* Linear projection of future positions
* Stationary gating to prevent unstable predictions

This approach provides a stable and computationally efficient baseline suitable for real-time systems.

## Project Structure

```
trajectory_forecast/
│
├── config.py        # Configuration system
├── drawing.py       # Visualization utilities
├── forecasting.py   # Velocity estimation and projection
├── tracker.py       # Track history management
├── inference.py     # Core pipeline
└── cli.py           # Command-line interface
└── utils.py         # For downloading assets from GitHub.
```

## Contributing

Contributions are welcome. If you would like to extend the forecasting models or improve tracking integration, please open an issue or submit a pull request.
