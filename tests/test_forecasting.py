import sys
import unittest

import numpy as np

from tf.forecasting import KalmanTrajectory
from tf.tracker import TrackManager


class ForecastTests(unittest.TestCase):
    def test_forecast_uses_velocity_and_acceleration(self):
        kf = KalmanTrajectory(10, 20, 1 / 10, 1.0, 10.0)
        kf.state[:] = [10, 20, -12, 30, 4, -6]
        expected = []
        for step in range(1, 8):
            t = step / 10
            expected.append((10 - 12 * t + 2 * t * t, 20 + 30 * t - 3 * t * t))
        np.testing.assert_allclose(kf.forecast(7), expected, rtol=0, atol=1e-5)
        self.assertEqual(kf.forecast(0), [])

    def test_filter_learns_acceleration_from_motion(self):
        kf = KalmanTrajectory(0, 0, 0.1, 2.0, 1.0)
        for frame in range(1, 30):
            t = frame * 0.1
            kf.predict()
            kf.update(20 * t + 3 * t * t, 5 * t)
        ax, ay = kf.acceleration()
        self.assertGreater(ax, 1.0)
        self.assertLess(abs(ay), 1.0)

    def test_short_gap_retains_history_and_advances_each_frame(self):
        manager = TrackManager(10, 20, 1.0, 10.0, max_gap_frames=2)
        manager.update(7, 100, 100)
        manager.cleanup({7})
        kf = manager.update(7, 103, 100)
        manager.cleanup({7})

        expected = KalmanTrajectory(0, 0, 1 / 20, 1.0, 10.0)
        expected.state = kf.state.copy()
        expected.P = kf.P.copy()

        manager.cleanup(set())
        manager.cleanup(set())
        self.assertIs(manager.filters[7], kf)
        for _ in range(3):
            expected.predict()
        expected.update(112, 100)

        same_filter = manager.update(7, 112, 100)
        self.assertIs(same_filter, kf)
        np.testing.assert_allclose(kf.state, expected.state)
        self.assertEqual(len(manager.history[7]), 3)

    def test_expired_tracks_release_filter_and_history(self):
        manager = TrackManager(10, 20, 1.0, 10.0, max_gap_frames=2)
        manager.update(7, 100, 100)
        manager.cleanup({7})
        for _ in range(3):
            manager.cleanup(set())
        self.assertNotIn(7, manager.filters)
        self.assertNotIn(7, manager.history)
        self.assertNotIn(7, manager.missed)

    def test_importing_forecast_core_does_not_import_video_stack(self):
        self.assertNotIn("tf.inference", sys.modules)


if __name__ == "__main__":
    unittest.main()
