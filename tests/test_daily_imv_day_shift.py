import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from combine_daily_imv_outputs_hourly_streamflow_hourly_eddy import (
	interpolate_daily_imv_to_hourly,
	merge_hourly_sources,
)


class DailyImvDayShiftTests(unittest.TestCase):
	def setUp(self):
		self.daily_imv = pd.DataFrame(
			{
				"date": pd.to_datetime(["2001-01-01", "2001-01-02", "2001-01-03"]),
				"pred_PET": [10.0, 20.0, 30.0],
				"obs_PET": [11.0, 21.0, 31.0],
			}
		)
		self.hourly_timestamps = pd.Series(
			pd.date_range("2001-01-01", "2001-01-03 23:00:00", freq="h")
		)

	def test_day_shift_zero_preserves_same_day_behavior(self):
		actual = interpolate_daily_imv_to_hourly(
			self.daily_imv,
			self.hourly_timestamps,
			"hist_scaled",
		)

		self.assertEqual(len(actual), 72)
		self.assertEqual(actual.loc[0, "pred_PET"], 10.0)
		self.assertEqual(actual.loc[23, "pred_PET"], 10.0)
		self.assertEqual(actual.loc[24, "pred_PET"], 20.0)
		self.assertNotIn("obs_PET", actual.columns)

	def test_day_shift_one_uses_previous_day_prediction(self):
		actual = interpolate_daily_imv_to_hourly(
			self.daily_imv,
			self.hourly_timestamps,
			"hist_scaled",
			day_shift=1,
		)

		self.assertEqual(len(actual), 48)
		self.assertEqual(actual.loc[0, "Datetime"], pd.Timestamp("2001-01-02 00:00:00"))
		self.assertEqual(actual.loc[0, "pred_PET"], 10.0)
		self.assertEqual(actual.loc[23, "pred_PET"], 10.0)
		self.assertEqual(actual.loc[24, "pred_PET"], 20.0)

	def test_day_shift_two_uses_prediction_from_two_days_earlier(self):
		actual = interpolate_daily_imv_to_hourly(
			self.daily_imv,
			self.hourly_timestamps,
			"hist_scaled",
			day_shift=2,
		)

		self.assertEqual(len(actual), 24)
		self.assertEqual(actual.loc[0, "Datetime"], pd.Timestamp("2001-01-03 00:00:00"))
		self.assertTrue((actual["pred_PET"] == 10.0).all())

	def test_explicit_day_shift_zero_matches_default_exactly(self):
		default = interpolate_daily_imv_to_hourly(
			self.daily_imv,
			self.hourly_timestamps,
			"hist_scaled",
		)
		explicit = interpolate_daily_imv_to_hourly(
			self.daily_imv,
			self.hourly_timestamps,
			"hist_scaled",
			day_shift=0,
		)

		assert_frame_equal(default, explicit)

	def test_negative_day_shift_is_rejected(self):
		with self.assertRaisesRegex(ValueError, "greater than or equal to 0"):
			interpolate_daily_imv_to_hourly(
				self.daily_imv,
				self.hourly_timestamps,
				"hist_scaled",
					day_shift=-1,
				)

	def test_observed_imvs_are_lagged_but_target_and_weather_are_current(self):
		timestamps = pd.Series(pd.date_range("2001-01-01", periods=48, freq="h"))
		hourly_flow = pd.DataFrame(
			{"Datetime": timestamps, "streamflow": range(100, 148)}
		)
		hourly_eddev = pd.DataFrame(
			{
				"Datetime": timestamps,
				"PET": range(48),
				"ET": range(200, 248),
				"T2": range(300, 348),
			}
		)
		hourly_imv = pd.DataFrame(
			{
				"Datetime": timestamps.iloc[24:].reset_index(drop=True),
				"scenario": "hist_scaled",
				"pred_streamflow": 999.0,
			}
		)

		actual = merge_hourly_sources(
			hourly_flow,
			hourly_eddev,
			hourly_imv,
			"hist_scaled",
			day_shift=1,
		)

		self.assertEqual(len(actual), 24)
		self.assertEqual(actual.loc[0, "Datetime"], pd.Timestamp("2001-01-02 00:00:00"))
		self.assertEqual(actual.loc[0, "PET"], 0)
		self.assertEqual(actual.loc[0, "ET"], 200)
		self.assertEqual(actual.loc[0, "streamflow"], 124)
		self.assertEqual(actual.loc[0, "T2"], 324)


if __name__ == "__main__":
	unittest.main()
