from dataclasses import dataclass
from typing import List, Optional, Tuple

import yaml


@dataclass
class ForecastConfig:
    """Tunable settings for detection, tracking, forecasting and drawing.

    Visual sizes left as ``None`` are filled in from the video resolution, so
    output looks consistent on any input while any value can still be overridden.
    """

    # Detection and tracking.
    conf: float = 0.4
    tracker: str = "bytetrack.yaml"
    classes: Optional[List[int]] = None

    # Trajectory and forecasting.
    history: int = 30
    min_points: int = 5
    forecast_steps: int = 35
    min_speed: float = 1.0
    process_noise: float = 1.0
    measurement_noise: float = 10.0

    # Drawing (auto-scaled from resolution when left as None).
    forecast_color: Tuple[int, int, int] = (108, 27, 255)
    line_thickness: Optional[int] = None
    forecast_thickness: Optional[int] = None
    forecast_radius: Optional[int] = None
    font_scale: Optional[float] = None
    font_thickness: Optional[int] = None
    padding: Optional[int] = None

    def __post_init__(self):
        """Apply the default class list when none is given."""
        if self.classes is None:
            self.classes = [0, 2, 5, 6, 7]

    @classmethod
    def from_yaml(cls, path: str) -> "ForecastConfig":
        """Build a config from a YAML file.

        Args:
            path (str): Path to the YAML config file.

        Returns:
            ForecastConfig: The loaded configuration.
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def resolve_visuals(self, width: int, height: int) -> None:
        """Fill any unset drawing sizes based on the video resolution.

        Sizes scale with the frame diagonal relative to 1080p, so lines and text
        stay readable on both small and 4K videos. Values set by the user are
        left unchanged.

        Args:
            width (int): Frame width in pixels.
            height (int): Frame height in pixels.
        """
        diagonal = (width ** 2 + height ** 2) ** 0.5
        reference = (1920 ** 2 + 1080 ** 2) ** 0.5
        scale = min(max(diagonal / reference, 0.5), 3.0)

        if self.line_thickness is None:
            self.line_thickness = max(1, round(2 * scale))
        if self.forecast_thickness is None:
            self.forecast_thickness = max(1, round(2 * scale))
        if self.forecast_radius is None:
            self.forecast_radius = max(2, round(6 * scale))
        if self.font_scale is None:
            self.font_scale = round(1.2 * scale, 2)
        if self.font_thickness is None:
            self.font_thickness = max(1, round(3 * scale))
        if self.padding is None:
            self.padding = max(2, round(8 * scale))
