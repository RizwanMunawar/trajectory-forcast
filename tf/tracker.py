from collections import defaultdict, deque

from .forecasting import KalmanTrajectory


class TrackManager:
    """Stores a Kalman filter and a position history for every tracked object."""

    def __init__(self, history_size, fps, process_noise, measurement_noise):
        """Set up empty history and filter stores.

        Args:
            history_size (int): Max number of past points kept per track.
            fps (float): Video frame rate, used as the filter time step.
            process_noise (float): Kalman process noise (motion flexibility).
            measurement_noise (float): Kalman measurement noise (detection trust).
        """
        self.history = defaultdict(lambda: deque(maxlen=history_size))
        self.filters = {}
        self.dt = 1.0 / fps
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

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
            kf.predict()
            kf.update(cx, cy)

        self.history[track_id].append(kf.position())
        return kf

    def cleanup(self, active_ids):
        """Drop history and filters for objects no longer being tracked.

        Args:
            active_ids (set[int]): Track ids seen in the current frame.
        """
        for tid in list(self.history.keys()):
            if tid not in active_ids:
                self.history.pop(tid, None)
                self.filters.pop(tid, None)
