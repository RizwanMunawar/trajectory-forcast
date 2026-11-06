import sys
import unittest

import numpy as np

from tf.config import ForecastConfig
from tf.forecasting import KalmanTrajectory
from tf.tracker import TrackManager


class ForecastTests(unittest.TestCase):
    def test_inference_performance_options_preserve_defaults(self):
        config = ForecastConfig()
        self.assertIsNone(config.imgsz)
        self.assertFalse(config.half)
        self.assertIsNone(config.device)
        self.assertEqual(config.confidence_floor, 0.1)
        self.assertEqual(config.confidence_exponent, 1.0)

        tuned = ForecastConfig(
            imgsz=480,
            half=True,
            device="0",
            confidence_floor=0.2,
            confidence_exponent=2.0,
        )
        self.assertEqual(tuned.imgsz, 480)
        self.assertTrue(tuned.half)
        self.assertEqual(tuned.device, "0")
        self.assertEqual(tuned.confidence_floor, 0.2)
        self.assertEqual(tuned.confidence_exponent, 2.0)

    def test_high_confidence_detection_is_trusted_more(self):
        high = KalmanTrajectory(0, 0, 0.1, 1.0, 10.0)
        low = KalmanTrajectory(0, 0, 0.1, 1.0, 10.0)
        high.P = np.eye(6, dtype=np.float32)
        low.P = np.eye(6, dtype=np.float32)

        high.update(10, 0, confidence=0.9)
        low.update(10, 0, confidence=0.2)

        self.assertGreater(high.position()[0], low.position()[0])
        self.assertGreater(high.P[0, 0], 0.0)
        self.assertGreater(low.P[0, 0], 0.0)

    def test_confidence_floor_prevents_unbounded_noise(self):
        floored = KalmanTrajectory(
            0, 0, 0.1, 1.0, 10.0, confidence_floor=0.1
        )
        at_floor = KalmanTrajectory(
            0, 0, 0.1, 1.0, 10.0, confidence_floor=0.1
        )
        floored.P = np.eye(6, dtype=np.float32)
        at_floor.P = np.eye(6, dtype=np.float32)
        floored.update(10, 0, confidence=0.0)
        at_floor.update(10, 0, confidence=0.1)
        np.testing.assert_allclose(floored.state, at_floor.state)
        np.testing.assert_allclose(floored.P, at_floor.P)

    def test_confidence_can_be_disabled_for_fixed_noise(self):
        fixed_a = KalmanTrajectory(
            0, 0, 0.1, 1.0, 10.0, confidence_exponent=0.0
        )
        fixed_b = KalmanTrajectory(
            0, 0, 0.1, 1.0, 10.0, confidence_exponent=0.0
        )
        fixed_a.update(10, 0, confidence=0.9)
        fixed_b.update(10, 0, confidence=0.1)
        np.testing.assert_allclose(fixed_a.state, fixed_b.state)
        np.testing.assert_allclose(fixed_a.P, fixed_b.P)

    def test_invalid_confidence_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            ForecastConfig(confidence_floor=0.0)
        with self.assertRaises(ValueError):
            ForecastConfig(confidence_exponent=-1.0)

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
        kf = manager.update(7, 103, 100, confidence=0.8)
        manager.cleanup({7})

        expected = KalmanTrajectory(0, 0, 1 / 20, 1.0, 10.0)
        expected.state = kf.state.copy()
        expected.P = kf.P.copy()

        manager.cleanup(set())
        manager.cleanup(set())
        self.assertIs(manager.filters[7], kf)
        for _ in range(3):
            expected.predict()
        expected.update(112, 100, confidence=0.8)

        same_filter = manager.update(7, 112, 100, confidence=0.8)
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
