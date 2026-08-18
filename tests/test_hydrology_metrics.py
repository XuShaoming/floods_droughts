import math
import unittest

import numpy as np

from hydrology_metrics import METRIC_NAMES, calculate_hydrology_metrics


class HydrologyMetricsTests(unittest.TestCase):
    def test_perfect_predictions(self):
        metrics = calculate_hydrology_metrics([1, 2, 3], [1, 2, 3])

        self.assertEqual(set(metrics), set(METRIC_NAMES))
        for name in ("R2", "NSE", "KGE", "KGE_r", "KGE_alpha", "KGE_beta"):
            self.assertAlmostEqual(metrics[name], 1.0)
        for name in ("MSE", "RMSE", "MAE", "MAPE", "NRMSE_sigma", "Bias"):
            self.assertAlmostEqual(metrics[name], 0.0)

    def test_kge_components_for_scaled_predictions(self):
        metrics = calculate_hydrology_metrics([2, 4, 6], [1, 2, 3])

        self.assertAlmostEqual(metrics["KGE_r"], 1.0)
        self.assertAlmostEqual(metrics["KGE_alpha"], 2.0)
        self.assertAlmostEqual(metrics["KGE_beta"], 2.0)
        self.assertAlmostEqual(metrics["KGE"], 1.0 - math.sqrt(2.0))
        self.assertAlmostEqual(metrics["Bias"], 2.0)

    def test_nse_and_sigma_normalized_rmse(self):
        metrics = calculate_hydrology_metrics([2, 3, 4], [1, 2, 3])

        self.assertAlmostEqual(metrics["NSE"], -0.5)
        self.assertAlmostEqual(metrics["R2"], metrics["NSE"])
        self.assertAlmostEqual(metrics["NRMSE_sigma"], math.sqrt(1.5))

    def test_nonfinite_pairs_are_omitted(self):
        metrics = calculate_hydrology_metrics(
            [1, np.nan, 3, np.inf],
            [1, 2, 3, 4],
        )

        self.assertAlmostEqual(metrics["RMSE"], 0.0)
        self.assertAlmostEqual(metrics["NSE"], 1.0)

    def test_undefined_metrics_are_nan_for_constant_observations(self):
        metrics = calculate_hydrology_metrics([2, 2, 2], [1, 1, 1])

        for name in ("R2", "NSE", "KGE", "KGE_r", "KGE_alpha", "NRMSE_sigma"):
            self.assertTrue(math.isnan(metrics[name]))
        self.assertAlmostEqual(metrics["KGE_beta"], 2.0)
        self.assertAlmostEqual(metrics["Bias"], 1.0)

    def test_mismatched_shapes_raise(self):
        with self.assertRaises(ValueError):
            calculate_hydrology_metrics([1, 2], [1])


if __name__ == "__main__":
    unittest.main()
