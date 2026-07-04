import cv2
import numpy as np


def draw_polyline(frame, pts, color, thickness=2):
    """Draw a connected line through a list of points.

    Args:
        frame (np.ndarray): Image to draw on.
        pts (list[tuple[int, int]]): Points to connect.
        color (tuple[int, int, int]): Line color in BGR.
        thickness (int): Line thickness in pixels.
    """
    if len(pts) < 2:
        return
    p = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [p], False, color, thickness, cv2.LINE_AA)


def draw_forecast(frame, pts, color, thickness=2, radius=6):
    """Draw the forecast path as a line with a few marker dots.

    Args:
        frame (np.ndarray): Image to draw on.
        pts (list[tuple[int, int]]): Forecast points.
        color (tuple[int, int, int]): Draw color in BGR.
        thickness (int): Forecast line thickness in pixels.
        radius (int): Marker dot radius in pixels.
    """
    if len(pts) < 2:
        return
    draw_polyline(frame, pts, color, thickness)
    for (x, y) in pts[:: max(1, len(pts) // 5)]:
        cv2.circle(frame, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)


def clamp_points(pts, w, h):
    """Keep only points inside the frame and cast them to ints.

    Args:
        pts (list[tuple[float, float]]): Points to filter.
        w (int): Frame width in pixels.
        h (int): Frame height in pixels.

    Returns:
        list[tuple[int, int]]: Points that fall within the frame.
    """
    return [(int(x), int(y)) for x, y in pts if 0 <= x < w and 0 <= y < h]
