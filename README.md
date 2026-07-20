# Trajectory Forecast

😍😍😍 **You can use any Ultralytics-supported model here.** 😍😍😍

[![CI](https://img.shields.io/github/actions/workflow/status/RizwanMunawar/trajectory-forcast/ci.yml?branch=main&logo=githubactions&logoColor=white)](https://github.com/RizwanMunawar/trajectory-forcast/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/trajectory-forecast?logo=pypi&logoColor=white)](https://pypi.org/project/trajectory-forecast/)
[![Downloads](https://static.pepy.tech/personalized-badge/trajectory-forecast?period=total&units=INTERNATIONAL_SYSTEM&left_color=black&right_color=gray&left_text=downloads)](https://pepy.tech/projects/trajectory-forecast)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-3776AB?logo=python&logoColor=green)
![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4.0%2B-00FFFF?logo=ultralytics&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO26-Object%20Tracking-FF6F00?logo=yolo&logoColor=white)
![Visitors](https://visitor-badge.laobi.icu/badge?page_id=RizwanMunawar.trajectory-forcast)
[![Blog](https://img.shields.io/badge/Blog-Trajectory_Forecasting-7B2CBF?logo=readthedocs&logoColor=white)](https://www.rizwanai.com/blog/object-tracking-and-trajectory-forecasting-with-yolo26)

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
trajectory-forecast --model yolo26n.pt \
                    --source "https://tinyurl.com/2f3yrppv" \
                    --output result.mp4 --show --save
```

If you want to adjust tracking and forecasting configuration, create a `config.yaml` in the directory and paste the mentioned content:

```yaml
# object detection confidence threshold
conf: 0.5

# tracker selection, i.e., "botsort.yaml" | "bytetrack.yaml"
# "ocsort.yaml", "deepocsort.yaml", "fasttrack.yaml", "tracktrack.yaml"                
tracker: "bytetrack.yaml"

# classes for object detection
classes: [2, 3, 5]

# store tracking history for the number of frames        
history: 40

# minimum tracking history to start calculating forecasting                 
min_points: 8

# total steps for forecasting; larger values extend the prediction horizon.             
forecast_steps: 35

# minimum speed (px/sec) before a forecast is drawn; filters out standing objects.
min_speed: 1.0

# Kalman motion flexibility; higher reacts faster, lower is smoother.
process_noise: 1.0

# Kalman detection trust; higher smooths harder (more noise assumed).
measurement_noise: 10.0

# Forecast point color (B, G, R)           
forecast_color: [255, 0, 0]

# Drawing sizes below are optional. If left out, they auto-scale to the video
# resolution. Set any of them to override.
# line_thickness: 2       # tracking box + trail thickness
# forecast_thickness: 2   # forecast line thickness
# forecast_radius: 6      # forecast marker dot radius
# font_scale: 1.2         # label text size
# font_thickness: 3       # label text thickness
# padding: 8              # label box padding
```

After that, you can run the code using the command mentioned below.

```bash
trajectory-forecast --model yolo26n.pt \
                    --source "https://tinyurl.com/2f3yrppv" \
                    --config "path/to/config.yaml"
```

### Python

```python
from tf import run_inference
from tf.config import ForecastConfig

config = ForecastConfig(
              conf=0.5,
              forecast_steps=50,
              measurement_noise=10.0,
              classes=[0, 2, 5, 6, 7]
            )

run_inference(
          model_path="yolo26s.pt",
          source="https://tinyurl.com/2f3yrppv",
          output_path="output.mp4",
          config=config
        )
```

## Forecasting methodology


Each tracked object is smoothed with a **constant-velocity Kalman filter**:

* The filter keeps a running estimate of position and velocity for every track.
* On each frame it predicts the next state, then corrects it with the new detection.
* Future positions are forecast by rolling that motion model forward `forecast_steps` frames.
* Objects slower than `min_speed` are skipped so standing targets don't get a forecast.

The earlier version estimated velocity by differencing adjacent frames
(`(p[i] - p[i-1]) / dt`), which divides a tiny per-frame delta by `dt = 1/fps` and so amplifies
detection noise by a factor of `fps` — the main source of forecast jitter. The Kalman filter
instead weighs each noisy detection against the predicted motion, so the estimated velocity, and
therefore the forecast, stays steady. On a straight, constant-velocity track with noisy
detections this cut the frame-to-frame movement of the forecast endpoint by roughly **30×**.

Two knobs control the smoothing: `measurement_noise` (how much detections are trusted; higher
smooths harder) and `process_noise` (how quickly the motion is allowed to change; higher reacts
faster). The filter also keeps predicting through short detection gaps, which helps during brief
occlusions.

## Project structure

<img width="1514" height="633" alt="high-level component structure image" src="https://github.com/user-attachments/assets/5f209bc9-9874-45b2-bd4e-1d0e160ffdbb" />


```markdown
tf/
│
├── config.py        # Configuration and resolution-based auto-scaling
├── drawing.py       # Visualization utilities
├── forecasting.py   # Kalman filter and forecasting
├── tracker.py       # Per-track filter and history management
├── inference.py     # Core pipeline
└── cli.py           # Command-line interface
└── utils.py         # For downloading assets from GitHub.
```

## Contributing

The contributions are always welcome. If you would like to extend the forecasting models or improve tracking integration, please open an [issue](https://github.com/RizwanMunawar/trajectory-forcast/issues/new) or submit a [pull request](https://github.com/RizwanMunawar/trajectory-forcast/pulls).
