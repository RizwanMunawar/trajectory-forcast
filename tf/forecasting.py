import numpy as np

def estimate_velocity(pts, fps, window=8):
    n = len(pts)
    if n < 2:
        return 0.0, 0.0

    k = min(window, n - 1)
    dt = 1.0 / fps
    vxs, vys = [], []

    for i in range(n - k, n):
        x1, y1 = pts[i - 1]
        x2, y2 = pts[i]
        vxs.append((x2 - x1) / dt)
        vys.append((y2 - y1) / dt)

    return float(np.median(vxs)), float(np.median(vys))


def forecast_points(last_xy, vx, vy, fps, steps):
    dt = 1.0 / fps
    lx, ly = last_xy
    return [(lx + vx * dt * s, ly + vy * dt * s)
            for s in range(1, steps + 1)]