import unittest

import numpy as np
import pandas as pd

from compare_flow_extremes import (
    bootstrap_ci,
    evaluate_events,
    paired_bootstrap_difference,
)


class CompareFlowExtremesTests(unittest.TestCase):
    def test_bootstrap_ci_is_reproducible(self):
        first = bootstrap_ci([1, 2, 3, 4], 500, np.random.default_rng(42))
        second = bootstrap_ci([1, 2, 3, 4], 500, np.random.default_rng(42))
        self.assertEqual(first, second)
        self.assertEqual(first[0], 2.5)
        self.assertLessEqual(first[1], first[0])
        self.assertGreaterEqual(first[2], first[0])

    def test_paired_difference_preserves_pairing(self):
        left = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
        right = pd.Series([2.0, 3.0, 4.0], index=["a", "b", "c"])
        estimate, low, high, win_rate = paired_bootstrap_difference(
            left, right, 500, np.random.default_rng(42), "median"
        )
        self.assertEqual(estimate, -1.0)
        self.assertEqual(low, -1.0)
        self.assertEqual(high, -1.0)
        self.assertEqual(win_rate, 1.0)

    def test_missing_predicted_events_score_zero_f1(self):
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2000-01-01", periods=12, freq="h"),
                "obs_streamflow": [2.0] * 12,
                "pred_streamflow": [0.0] * 12,
            }
        )
        summary, _, _, _ = evaluate_events(
            frame,
            "Test",
            "streamflow",
            threshold=1.0,
            gap_hours=0.0,
            min_event_hours=6.0,
            match_window_hours=24.0,
        )
        self.assertEqual(summary["observed_event_count"], 1)
        self.assertEqual(summary["predicted_event_count"], 0)
        self.assertEqual(summary["event_precision"], 0.0)
        self.assertEqual(summary["event_recall"], 0.0)
        self.assertEqual(summary["event_F1"], 0.0)


if __name__ == "__main__":
    unittest.main()
