import numpy as np


class KalmanTrajectory:
    """Constant-velocity Kalman filter for a single tracked object.

    Keeps a smoothed estimate of position and velocity so the forecast stays
    steady even when detections are noisy, and predicts future points by rolling
    the motion model forward.
    """

    def __init__(self, x, y, dt, process_noise, measurement_noise):
        """Create a filter initialised at the object's first position.

        Args:
            x (float): Initial x position in pixels.
            y (float): Initial y position in pixels.
            dt (float): Time between frames in seconds (1 / fps).
            process_noise (float): How much the motion is allowed to change;
                higher reacts faster, lower is smoother.
            measurement_noise (float): How noisy detections are assumed to be;
                higher trusts the model more and smooths harder.
        """
        self.dt = dt

        # State vector: [x, y, vx, vy].
        self.state = np.array([x, y, 0.0, 0.0], dtype=np.float64)

        # State transition (constant velocity) and measurement matrices.
        self.F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)

        # Covariances: P starts uncertain, Q is process noise, R is measurement noise.
        self.P = np.eye(4, dtype=np.float64) * 1000.0
        self.Q = np.eye(4, dtype=np.float64) * process_noise
        self.R = np.eye(2, dtype=np.float64) * measurement_noise

    def predict(self):
        """Advance the state one frame forward using the motion model."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, x, y):
        """Correct the state with a new detection.

        Args:
            x (float): Measured x position in pixels.
            y (float): Measured y position in pixels.
        """
        z = np.array([x, y], dtype=np.float64)
        residual = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ residual
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def position(self):
        """Return the smoothed (x, y) position in pixels."""
        return float(self.state[0]), float(self.state[1])

    def velocity(self):
        """Return the estimated (vx, vy) velocity in pixels per second."""
        return float(self.state[2]), float(self.state[3])

    def forecast(self, steps):
        """Predict future positions by rolling the motion model forward.

        Args:
            steps (int): Number of frames to predict ahead.

        Returns:
            list[tuple[float, float]]: Predicted (x, y) points.
        """
        state = self.state.copy()
        points = []
        for _ in range(steps):
            state = self.F @ state
            points.append((float(state[0]), float(state[1])))
        return points
