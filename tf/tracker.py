from collections import defaultdict, deque

class TrackManager:
    def __init__(self, history_size: int, ema_alpha: float):
        self.history = defaultdict(lambda: deque(maxlen=history_size))
        self.last_smooth = {}
        self.ema_alpha = ema_alpha

    def update(self, track_id, cx, cy):
        if track_id in self.last_smooth:
            lx, ly = self.last_smooth[track_id]
            sx = self.ema_alpha * cx + (1 - self.ema_alpha) * lx
            sy = self.ema_alpha * cy + (1 - self.ema_alpha) * ly
        else:
            sx, sy = cx, cy

        self.last_smooth[track_id] = (sx, sy)
        self.history[track_id].append((sx, sy))

    def cleanup(self, active_ids):
        for tid in list(self.history.keys()):
            if tid not in active_ids:
                self.history.pop(tid, None)
                self.last_smooth.pop(tid, None)