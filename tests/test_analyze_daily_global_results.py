"""Focused tests for the daily multi-watershed analysis helpers."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_daily_global_results import (
    EXPECTED_FEATURES,
    ExperimentSpec,
    evaluate_frame,
    load_reconstruction,
    summarize_kge,
)


class DailyGlobalAnalysisTests(unittest.TestCase):
    def test_summary_uses_unweighted_mean_and_sample_standard_deviation(self):
        spec = ExperimentSpec("daily_global_PET", Path("unused"), "PET", EXPECTED_FEATURES)
        records = []
        for split in ("train", "val", "test"):
            records.extend(
                [
                    {"target": "PET", "split": split, "watershed": "A", "KGE": 0.6},
                    {"target": "PET", "split": split, "watershed": "B", "KGE": 1.0},
                ]
            )

        summary = summarize_kge(pd.DataFrame(records), [spec]).iloc[0]

        self.assertAlmostEqual(summary["train_kge_mean"], 0.8)
        self.assertAlmostEqual(summary["train_kge_std"], np.std([0.6, 1.0], ddof=1))
        self.assertEqual(summary["train_n_watersheds"], 2)
        self.assertAlmostEqual(summary["valid_kge_mean"], 0.8)
        self.assertAlmostEqual(summary["test_kge_mean"], 0.8)

    def test_evaluate_frame_omits_nonfinite_pairs_and_preserves_perfect_skill(self):
        spec = ExperimentSpec("daily_global_PET", Path("unused"), "PET", EXPECTED_FEATURES)
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2001-01-01", periods=4, freq="D"),
                "pred_PET": [1.0, 2.0, np.nan, 4.0],
                "obs_PET": [1.0, 2.0, 3.0, 4.0],
                "scenario": ["hist_scaled"] * 4,
            }
        )

        row = evaluate_frame(frame, spec, "test", "BlueEarth", Path("source.csv"))

        self.assertEqual(row["n_pairs"], 3)
        self.assertEqual(row["n_dropped_nonfinite"], 1)
        self.assertAlmostEqual(row["KGE"], 1.0)
        self.assertAlmostEqual(row["NSE"], 1.0)
        self.assertAlmostEqual(row["PBIAS_pct"], 0.0)

    def test_load_reconstruction_rejects_duplicate_timestamp_scenario_rows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "duplicate.csv"
            pd.DataFrame(
                {
                    "timestamp": ["2001-01-01", "2001-01-01"],
                    "pred_PET": [1.0, 1.1],
                    "obs_PET": [1.0, 1.0],
                    "scenario": ["hist_scaled", "hist_scaled"],
                }
            ).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "duplicate timestamp/scenario"):
                load_reconstruction(path, "PET")


if __name__ == "__main__":
    unittest.main()
