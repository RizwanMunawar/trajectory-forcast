# Trajectory Forecast

😍😍😍 **You can use any Ultralytics-supported model here.** 😍😍😍

![Ultralytics 8.4.0](https://img.shields.io/badge/Ultralytics-8.4.0%2B-blue?logo=ultralytics&logoColor=white) ![Python 3.10 | Python3.11 | Python 3.12 | 3.13 | 3.14](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue?logo=python&logoColor=white) ![Visitors](https://visitor-badge.laobi.icu/badge?page_id=RizwanMunawar.trajectory-forcast) [![PyPI Downloads](https://static.pepy.tech/personalized-badge/trajectory-forecast?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/trajectory-forecast) 


<img width="1112" height="584" alt="trajectory-forecast-usage-code-snippet" src="https://github.com/user-attachments/assets/75e3b90f-4e67-44b2-a793-390f94f66018"/><br>

Trajectory Forecast is a lightweight, modular extension built on top of Ultralytics YOLO that enables real-time multi-object 
tracking with future motion prediction. It combines detection, tracking, motion history modeling, and velocity-based forecasting 
into a unified pipeline that can be used both as a command-line tool and as a Python library. The system is designed for practical computer vision applications such as traffic analytics, surveillance systems, robotics pipelines, and edge AI deployments.

- [Installation](#installation)
- [Usage](#usage)
  - [CLI](#cli)
  - [Python](#python)
- [Forecasting methodology](#forecasting-methodology)
- [Project structure](#project-structure)
- [Contribute](#contributing)
  
https://github.com/user-attachments/assets/9a1267c2-4ba4-49f6-9802-e80fed5e682f

## Installation

```bash
pip install trajectory-forecast
```

## Usage

### CLI

Run tracking and forecasting on a video.

```bash
trajectory-forecast --model yolo26n.pt --source "https://tinyurl.com/bddswzba" --output result.mp4
```

If you want to adjust tracking and forecasting configuration, create a `config.yaml` in the directory and paste the mentioned content:

```yaml
conf: 0.5                   # object detection confidence threshold
tracker: "bytetrack.yaml"   # tracker selection, i.e., "botsort.yaml" or "bytetrack.yaml"
classes: [2, 3, 5]          # classes for object detection
history: 40                 # store tracking history for number of frames
min_points: 8               # minimum tracking history to start calculating forecasting
forecast_steps: 35          # total steps for forecasting, > 40 can cause gitter effect.
vel_window: 10              # previous frames used to estimate the object's velocity.
ema_alpha: 0.7              # used to smooth the velocity or trajectory prediction.
forecast_color: [255, 0, 0] # Forecast point color (B, G, R)
```

After that, you can run the code using the command mentioned below.

```bash
trajectory-forecast --model yolo26n.pt --source "https://tinyurl.com/bddswzba" --config "path/to/config.yaml"
```

### Python

```python
from tf import run_inference
from tf.config import ForecastConfig

config = ForecastConfig(conf=0.5, forecast_steps=50, ema_alpha=0.7, classes=[0, 2, 5, 6, 7])

run_inference(model_path="yolo26s.pt", source="https://tinyurl.com/2t2j2vs5", output_path="output.mp4", config=config)
```

## Forecasting methodology


The current forecasting implementation is based on:

* Exponential moving average smoothing of object centers
* Median velocity estimation over a sliding window
* Linear projection of future positions
* Stationary gating to prevent unstable predictions

This approach provides a stable and computationally efficient baseline suitable for real-time systems.

## Project structure

<img width="1514" height="633" alt="high-level component structure image" src="https://github.com/user-attachments/assets/5f209bc9-9874-45b2-bd4e-1d0e160ffdbb" />


```markdown
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

## Contributing

The contributions are always welcome. If you would like to extend the forecasting models or improve tracking integration, please open an [issue](https://github.com/RizwanMunawar/trajectory-forcast/issues/new) or submit a [pull request](https://github.com/RizwanMunawar/trajectory-forcast/pulls).
