# Trajectory Forecast

<p align="center">
  <a href="https://github.com/RizwanMunawar/trajectory-forcast/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/RizwanMunawar/trajectory-forcast/ci.yml?branch=main&logo=githubactions&logoColor=white" alt="CI"></a>
  <a href="https://pypi.org/project/trajectory-forecast/"><img src="https://img.shields.io/pypi/v/trajectory-forecast?logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://pepy.tech/projects/trajectory-forecast"><img src="https://static.pepy.tech/personalized-badge/trajectory-forecast?period=total&units=INTERNATIONAL_SYSTEM&left_color=black&right_color=gray&left_text=downloads" alt="Downloads"></a>
  <img src="https://img.shields.io/badge/Python-3.10--3.14-3776AB?logo=python&logoColor=white" alt="Python 3.10-3.14">
  <img src="https://img.shields.io/badge/Ultralytics-8.4.0%2B-00FFFF?logo=ultralytics&logoColor=white" alt="Ultralytics 8.4.0+">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=RizwanMunawar.trajectory-forcast" alt="Visitors">
  <a href="https://www.rizwanai.com/blog/object-tracking-and-trajectory-forecasting-with-yolo26"><img src="https://img.shields.io/badge/Blog-Trajectory_Forecasting-7B2CBF?logo=readthedocs&logoColor=white" alt="Trajectory Forecasting Blog"></a>
</p>

Real-time multi-object tracking with lightweight trajectory forecasting, built on top of Ultralytics YOLO.

Track objects in video, keep their motion history, and estimate where they are moving next. Trajectory Forecast can be used from the command line or directly from Python.

https://github.com/user-attachments/assets/9a1267c2-4ba4-49f6-9802-e80fed5e682f

## Installation

```bash
pip install trajectory-forecast
```

## Quick start

Run tracking and trajectory forecasting on a video:

```bash
trajectory-forecast \
  --model yolo26n.pt \
  --source "https://tinyurl.com/2f3yrppv" \
  --output result.mp4 \
  --show \
  --save
```

Any Ultralytics-supported detection model can be used.

### Python

```python
from tf import run_inference

run_inference(
    model_path="yolo26n.pt",
    source="video.mp4",
    output_path="result.mp4",
)
```

## Configuration

The defaults work without a config file. To customize tracking or forecasting, create a YAML file:

```yaml
conf: 0.5
tracker: "bytetrack.yaml"
classes: [0, 2, 5, 6, 7]

history: 30
min_points: 5
forecast_steps: 35
min_speed: 1.0
max_gap_frames: 5

process_noise: 1.0
measurement_noise: 10.0

forecast_color: [255, 0, 0]
```

Then pass it to the CLI:

```bash
trajectory-forecast \
  --model yolo26n.pt \
  --source "video.mp4" \
  --config config.yaml
```

The most useful forecasting options are:

- `forecast_steps` — how many future frames to predict.
- `min_points` — tracking history required before forecasting starts.
- `min_speed` — skips forecasts for nearly stationary objects.
- `max_gap_frames` — keeps motion state through short detection gaps.
- `process_noise` — controls how quickly the motion estimate adapts.
- `measurement_noise` — controls how strongly detection noise is smoothed.

## How forecasting works

Each tracked object has its own acceleration-aware Kalman motion state.

On every frame, the filter updates the object's position, velocity, and acceleration from the latest tracked position. Future points are then generated from that filtered motion state. This helps the forecast respond to changing speed instead of assuming that every object will continue at one fixed velocity.

Short detection gaps retain the motion state for up to `max_gap_frames`. If the tracker assigns a completely new ID, a new forecast state is created for that track.

The forecasting layer uses `float32` state arrays and vectorized future-point generation to keep its per-frame overhead small. The terminal also reports trajectory forecasting time separately from the detector's inference timing.

## Project structure

```text
tf/
├── cli.py           # Command-line interface
├── config.py        # Configuration
├── drawing.py       # Visualization
├── forecasting.py   # Motion model and future prediction
├── inference.py     # Detection, tracking and forecasting pipeline
├── tracker.py       # Per-track state and history
└── utils.py         # Utilities
```

## Contributing

Contributions and improvements are welcome. Open an issue or pull request if you find a bug or have an idea for improving tracking or forecasting.

For a practical walkthrough, see the [trajectory forecasting guide](https://www.rizwanai.com/blog/object-tracking-and-trajectory-forecasting-with-yolo26).
