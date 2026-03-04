# Trajectory Forecast

Trajectory Forecast is a lightweight, modular extension built on top of Ultralytics YOLO that enables real-time multi-object 
tracking with future motion prediction. It combines detection, tracking, motion history modeling, and velocity-based forecasting 
into a unified pipeline that can be used both as a command-line tool and as a Python library. The system is designed for practical computer vision applications such as traffic analytics, surveillance systems, robotics pipelines, and edge AI deployments.


## Installation

### Recommended (Development Mode)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### CLI (Command-Line)

Run tracking and forecasting on a video.

```bash
trajectory-forecast \
  --model yolo26n.pt \
  --source "https://tinyurl.com/bddswzba" \
  --output result.mp4
```

If you want to adjust tracking and forecasting configuration, create a `config.yaml` in directory and paste the mentioned content:

```yaml
conf: 0.5                   # object detection confidence threshold
tracker: "bytetrack.yaml"   # tracker selection i.e "botsort.yaml" or "bytetrack.yaml"
classes: [2, 3, 5]          # classes for object detection
history: 40                 # store tracking history for number of frames
min_points: 8               # minimum tracking history to start calculating forecasting
forecast_steps: 35          # total steps for forecasting, > 40 can cause gitter effect.
vel_window: 10              # previous frames used to estimate the object's velocity.
ema_alpha: 0.7              # used to smooth the velocity or trajectory prediction.
forecast_color: [255, 0, 0] # Forecast point color (B, G, R)
```

After that, you can use the code using the mentioned command below.

```bash
trajectory-forecast \
  --model yolo26n.pt \
  --source "https://tinyurl.com/bddswzba" \
  --config "path/to/config.yaml"
```

### Python

```python
from tf import run_inference
from tf.config import ForecastConfig

config = ForecastConfig(
    conf=0.5,
    forecast_steps=50,
    ema_alpha=0.7,
    classes=[0, 2, 5, 6, 7],
)

run_inference(
    model_path="yolo26s.pt",
    source="https://tinyurl.com/2t2j2vs5",
    output_path="output.mp4",
    config=config,
)
```

### Forecasting Methodology

The current forecasting implementation is based on:

* Exponential moving average smoothing of object centers
* Median velocity estimation over a sliding window
* Linear projection of future positions
* Stationary gating to prevent unstable predictions

This approach provides a stable and computationally efficient baseline suitable for real-time systems.

### Project Structure

```
tf/
│
├── config.py        # Configuration system
├── drawing.py       # Visualization utilities
├── forecasting.py   # Velocity estimation and projection
├── tracker.py       # Track history management
├── inference.py     # Core pipeline
└── cli.py           # Command-line interface
└── utils.py         # For downloading assets from GitHub.
```

### Contributing

Contributions are welcome. If you would like to extend the forecasting models or improve tracking integration, please open an issue or submit a pull request.
