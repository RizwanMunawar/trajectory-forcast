import sys
import unittest

import numpy as np

from tf.forecasting import KalmanTrajectory
from tf.tracker import TrackManager


class ForecastTests(unittest.TestCase):
    def test_forecast_matches_constant_velocity_transition(self):
        kf = KalmanTrajectory(10, 20, 1 / 30, 1.0, 10.0)
        kf.state[:] = [10, 20, -12, 33]
        future = kf.state.copy()
        expected = []
        for _ in range(35):
            future = kf.F @ future
            expected.append(tuple(future[:2]))
        np.testing.assert_allclose(kf.forecast(35), expected, rtol=0, atol=1e-12)
        self.assertEqual(kf.forecast(0), [])

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
