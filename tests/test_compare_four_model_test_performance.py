"""Tests for the paired four-model paper-table analysis."""

import unittest

import numpy as np
import pandas as pd

from compare_four_model_test_performance import (
    MODEL_ORDER,
    MODEL_STYLES,
    align_four_model_sources,
    evaluate_watershed,
    rank_formats,
)


class FourModelTestPerformanceTests(unittest.TestCase):
    def test_requested_model_order_preserves_model_colors(self):
        self.assertEqual(
            MODEL_ORDER,
            ("D-LSTM", "H-LSTM", "MR-STF", "MR-PTF"),
        )
        self.assertEqual(MODEL_STYLES["D-LSTM"][0], "#CC79A7")
        self.assertEqual(MODEL_STYLES["H-LSTM"][0], "#0072B2")
        self.assertEqual(MODEL_STYLES["MR-STF"][0], "#D55E00")
        self.assertEqual(MODEL_STYLES["MR-PTF"][0], "#009E73")

    def setUp(self):
        timestamps = pd.date_range("2001-01-01", periods=72, freq="h")
        hour = timestamps.hour.to_numpy(dtype=float)
        day = (timestamps.day - timestamps.day.min()).to_numpy(dtype=float)
        observation = 100.0 + 20.0 * day + hour
        self.h_lstm = pd.DataFrame(
            {
                "timestamp": timestamps,
                "scenario": "hist_scaled",
                "obs_streamflow": observation,
                "pred_streamflow": observation,
            }
        )
        self.mr_stf = self.h_lstm.assign(pred_streamflow=observation + 1.0)
        self.mr_ptf = self.h_lstm.assign(pred_streamflow=observation + 2.0)
        daily = self.h_lstm.assign(date=timestamps.floor("D")).groupby(
            "date", as_index=False
        )["obs_streamflow"].mean()
        self.d_lstm = daily.assign(
            timestamp=lambda frame: frame["date"],
            scenario="hist_scaled",
            pred_streamflow=lambda frame: frame["obs_streamflow"],
        )[
            [
                "timestamp",
                "date",
                "scenario",
                "obs_streamflow",
                "pred_streamflow",
            ]
        ]

    def test_alignment_uses_complete_shared_days_and_repeats_daily_values(self):
        aligned, audit = align_four_model_sources(
            self.h_lstm, self.mr_stf, self.mr_ptf, self.d_lstm, "Synthetic"
        )

        self.assertEqual(len(aligned), 72)
        self.assertEqual(audit["common_complete_days"], 3)
        self.assertTrue((aligned.groupby("date")["pred_d_lstm"].nunique() == 1).all())
        self.assertAlmostEqual(audit["daily_expansion_max_abs_difference"], 0.0)
        self.assertAlmostEqual(audit["mr_stf_reference_max_abs_difference"], 0.0)
        self.assertAlmostEqual(audit["mr_ptf_reference_max_abs_difference"], 0.0)

    def test_evaluation_returns_all_four_models_on_one_reference(self):
        aligned, _ = align_four_model_sources(
            self.h_lstm, self.mr_stf, self.mr_ptf, self.d_lstm, "Synthetic"
        )
        overall, extreme, _, _ = evaluate_watershed(
            aligned,
            "Synthetic",
            0.10,
            0.90,
            0.95,
            24.0,
            6.0,
            24.0,
        )
        metrics = pd.DataFrame(overall).set_index("model")

        self.assertEqual(len(overall), 4)
        self.assertEqual(len(extreme), 4)
        self.assertAlmostEqual(metrics.loc["H-LSTM", "RMSE"], 0.0)
        self.assertAlmostEqual(metrics.loc["MR-STF", "RMSE"], 1.0)
        self.assertAlmostEqual(metrics.loc["MR-PTF", "RMSE"], 2.0)
        self.assertGreater(metrics.loc["D-LSTM", "RMSE"], 0.0)

    def test_ranking_supports_lower_higher_and_closest_to_one(self):
        values = {"A": 0.8, "B": 1.1, "C": 0.6, "D": 1.4}

        lower = rank_formats(values, 1, "lower")
        higher = rank_formats(values, 1, "higher")
        closest = rank_formats(values, 1, "closest_one")

        self.assertEqual(lower["C"], "**0.6**")
        self.assertEqual(higher["D"], "**1.4**")
        self.assertEqual(closest["B"], "**1.1**")
        self.assertEqual(closest["A"], "<u>0.8</u>")


if __name__ == "__main__":
    unittest.main()
