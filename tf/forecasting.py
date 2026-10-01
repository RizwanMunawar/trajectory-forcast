import numpy as np


class KalmanTrajectory:
    """Constant-acceleration Kalman filter for one tracked object."""

    def __init__(self, x, y, dt, process_noise, measurement_noise):
        self.dt = float(dt)
        dt2 = self.dt * self.dt

        # State: [x, y, vx, vy, ax, ay].
        self.state = np.array([x, y, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.F = np.array(
            [
                [1, 0, self.dt, 0, 0.5 * dt2, 0],
                [0, 1, 0, self.dt, 0, 0.5 * dt2],
                [0, 0, 1, 0, self.dt, 0],
                [0, 0, 0, 1, 0, self.dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.H = np.array(
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=np.float32
        )
        self.P = np.eye(6, dtype=np.float32) * 1000.0

        # White-jerk process model. This lets velocity change smoothly instead
        # of forcing every target to continue forever at one fixed velocity.
        q = float(process_noise)
        dt3, dt4, dt5 = self.dt**3, self.dt**4, self.dt**5
        q1 = np.array(
            [
                [dt5 / 20, dt4 / 8, dt3 / 6],
                [dt4 / 8, dt3 / 3, dt2 / 2],
                [dt3 / 6, dt2 / 2, self.dt],
            ],
            dtype=np.float32,
        ) * q
        self.Q = np.zeros((6, 6), dtype=np.float32)
        ix, iy = (0, 2, 4), (1, 3, 5)
        self.Q[np.ix_(ix, ix)] = q1
        self.Q[np.ix_(iy, iy)] = q1
        self.R = np.eye(2, dtype=np.float32) * float(measurement_noise)
        self._I = np.eye(6, dtype=np.float32)

    def predict(self):
        """Advance the state by one frame."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, x, y):
        """Correct the motion state with a measured position."""
        z = np.array([x, y], dtype=np.float32)
        residual = z - self.H @ self.state
        ph_t = self.P @ self.H.T
        s = self.H @ ph_t + self.R

        # Solve the 2x2 innovation system instead of explicitly inverting it.
        k = np.linalg.solve(s, ph_t.T).T
        self.state += k @ residual

        # Joseph form keeps covariance numerically stable during long streams.
        ikh = self._I - k @ self.H
        self.P = ikh @ self.P @ ikh.T + k @ self.R @ k.T

    def position(self):
        return float(self.state[0]), float(self.state[1])

    def velocity(self):
        return float(self.state[2]), float(self.state[3])

    def acceleration(self):
        return float(self.state[4]), float(self.state[5])

    def forecast(self, steps):
        """Predict future positions with the current velocity and acceleration."""
        if steps <= 0:
            return []

        x, y, vx, vy, ax, ay = self.state
        t = self.dt * np.arange(1, steps + 1, dtype=np.float32)
        fx = x + vx * t + 0.5 * ax * t * t
        fy = y + vy * t + 0.5 * ay * t * t
        return list(zip(fx.astype(float), fy.astype(float)))
