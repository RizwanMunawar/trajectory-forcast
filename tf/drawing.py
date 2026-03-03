import cv2
import numpy as np

def draw_polyline(frame, pts, color, thickness=2):
    if len(pts) < 2:
        return
    p = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [p], False, color, thickness, cv2.LINE_AA)

def draw_forecast(frame, pts, color):
    if len(pts) < 2:
        return
    draw_polyline(frame, pts, color, 1)
    for (x, y) in pts[::max(1, len(pts)//5)]:
        cv2.circle(frame, (int(x), int(y)), 6, color, -1, cv2.LINE_AA)

def clamp_points(pts, w, h):
    return [(int(x), int(y)) for x, y in pts if 0 <= x < w and 0 <= y < h]