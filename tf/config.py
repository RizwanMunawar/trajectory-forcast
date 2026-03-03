from dataclasses import dataclass
from typing import List, Tuple
import yaml


@dataclass
class ForecastConfig:
    # Model settings
    conf: float = 0.4
    tracker: str = "bytetrack.yaml"
    classes: List[int] = None

    # Trajectory settings
    history: int = 30
    min_points: int = 5
    forecast_steps: int = 35
    vel_window: int = 8
    ema_alpha: float = 0.6

    # Drawing
    forecast_color: Tuple[int, int, int] = (108, 27, 255)
    font_scale: float = 1.2
    font_thickness: int = 4
    padding: int = 8

    def __post_init__(self):
        if self.classes is None:
            self.classes = [0, 2, 5, 6, 7]

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)