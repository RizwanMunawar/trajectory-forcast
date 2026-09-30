from collections import defaultdict, deque

from .forecasting import KalmanTrajectory


class TrackManager:
    """Stores a Kalman filter and a position history for every tracked object."""

    def __init__(
        self, history_size, fps, process_noise, measurement_noise, max_gap_frames=5
    ):
        """Set up empty history and filter stores.

        Args:
            history_size (int): Max number of past points kept per track.
            fps (float): Video frame rate, used as the filter time step.
            process_noise (float): Kalman process noise (motion flexibility).
            measurement_noise (float): Kalman measurement noise (detection trust).
            max_gap_frames (int): Missing frames to retain a track's state.
        """
        self.history = defaultdict(lambda: deque(maxlen=history_size))
        self.filters = {}
        self.missed = {}
        self.dt = 1.0 / fps
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        if max_gap_frames < 0:
            raise ValueError("max_gap_frames must be non-negative")
        self.max_gap_frames = max_gap_frames

    def update(self, track_id, cx, cy):
        """Feed a new detection into the track's filter and store the result.

        Args:
            track_id (int): Tracker id of the object.
            cx (float): Detected center x in pixels.
            cy (float): Detected center y in pixels.

        Returns:
            KalmanTrajectory: The track's filter, holding the smoothed state.
        """
        kf = self.filters.get(track_id)
        if kf is None:
            kf = KalmanTrajectory(
                cx, cy, self.dt, self.process_noise, self.measurement_noise
            )
            self.filters[track_id] = kf
        else:
            # Advance through every missing frame before correcting with the
            # detection. One predict() would leave the velocity a frame or
            # more behind after a short occlusion.
            for _ in range(self.missed[track_id] + 1):
                kf.predict()
            kf.update(cx, cy)

        self.missed[track_id] = 0
        self.history[track_id].append(kf.position())
        return kf

    def cleanup(self, active_ids):
        """Retain brief gaps, then drop stale history and filters.

        Args:
            active_ids (set[int]): Track ids seen in the current frame.
        """
        for tid in list(self.filters):
            if tid not in active_ids:
                self.missed[tid] += 1
                if self.missed[tid] > self.max_gap_frames:
                    self.history.pop(tid, None)
                    self.filters.pop(tid, None)
                    self.missed.pop(tid, None)
