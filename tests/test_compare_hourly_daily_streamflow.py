"""Focused tests for the hourly-versus-daily streamflow comparison."""

import unittest

import numpy as np
import pandas as pd

from compare_hourly_daily_streamflow import align_basin_sources, evaluate_basin


class HourlyDailyStreamflowComparisonTests(unittest.TestCase):
    def setUp(self):
        timestamps = pd.date_range("2001-01-01", periods=72, freq="h")
        hour = timestamps.hour.to_numpy(dtype=float)
        day = (timestamps.day - timestamps.day.min()).to_numpy(dtype=float)
        observation = 100.0 + 10.0 * day + hour
        self.hourly = pd.DataFrame(
            {
                "timestamp": timestamps,
                "scenario": "hist_scaled",
                "obs_streamflow": observation,
                "pred_streamflow": observation,
            }
        )
        self.mr_stf = self.hourly.assign(
            pred_streamflow=self.hourly["obs_streamflow"] + 1.0
        )
        hourly_daily_mean = self.hourly.assign(
            date=self.hourly["timestamp"].dt.floor("D")
        ).groupby("date", as_index=False)["obs_streamflow"].mean()
        self.daily = hourly_daily_mean.assign(
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

    def test_alignment_repeats_each_daily_prediction_for_24_hours(self):
        hourly_common, daily_common, audit = align_basin_sources(
            self.hourly, self.mr_stf, self.daily, "Synthetic"
        )

        self.assertEqual(len(hourly_common), 72)
        self.assertEqual(len(daily_common), 3)
        repeated_counts = hourly_common.groupby("date")["pred_d_lstm"].nunique()
        self.assertTrue((repeated_counts == 1).all())
        np.testing.assert_allclose(
            daily_common["pred_d_lstm"], self.daily["pred_streamflow"]
        )
        self.assertEqual(audit["common_complete_days"], 3)
        self.assertAlmostEqual(audit["daily_reference_max_abs_difference"], 0.0)
        self.assertAlmostEqual(
            audit["daily_prediction_expansion_max_abs_difference"], 0.0
        )
        self.assertAlmostEqual(audit["mr_stf_reference_max_abs_difference"], 0.0)

    def test_models_use_shared_reference_and_intraday_diagnostic_is_finite(self):
        hourly_common, daily_common, _ = align_basin_sources(
            self.hourly, self.mr_stf, self.daily, "Synthetic"
        )
        metric_rows, intraday_rows = evaluate_basin(
            "Synthetic", hourly_common, daily_common
        )
        metrics = pd.DataFrame(metric_rows).set_index(
            ["evaluation_scale", "model"]
        )
        intraday = pd.DataFrame(intraday_rows).set_index("model")

        self.assertAlmostEqual(metrics.loc[("hourly", "H-LSTM"), "RMSE"], 0.0)
        self.assertAlmostEqual(metrics.loc[("hourly", "H-LSTM"), "NSE"], 1.0)
        self.assertAlmostEqual(metrics.loc[("hourly", "H-LSTM"), "KGE"], 1.0)
        self.assertAlmostEqual(metrics.loc[("daily_mean", "D-LSTM"), "RMSE"], 0.0)
        self.assertAlmostEqual(metrics.loc[("hourly", "MR-STF"), "RMSE"], 1.0)
        self.assertGreater(metrics.loc[("hourly", "D-LSTM"), "RMSE"], 0.0)
        self.assertAlmostEqual(
            intraday.loc["H-LSTM", "intraday_anomaly_RMSE"], 0.0
        )
        self.assertGreater(
            intraday.loc["D-LSTM", "intraday_anomaly_RMSE"], 0.0
        )
        self.assertTrue(
            np.isnan(intraday.loc["D-LSTM", "intraday_anomaly_correlation"])
        )

    def test_alignment_rejects_incomplete_calendar_days(self):
        incomplete = self.hourly.iloc[:-1].copy()

        with self.assertRaisesRegex(ValueError, "without exactly 24 hours"):
            align_basin_sources(incomplete, self.mr_stf, self.daily, "Synthetic")


if __name__ == "__main__":
    unittest.main()
