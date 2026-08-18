#!/usr/bin/env python3
"""
Combine daily IMV outputs with hourly streamflow and hourly EDDEV data.

Workflow per watershed and split:
1. Load daily IMV outputs from experiments/data_processed/{watershed}_{split}_imv_outputs.csv
2. For each scenario in the IMV file, load hourly flow and EDDEV files.
3. Interpolate daily IMV values to hourly resolution. By default, each daily
   value is assigned to all hourly timestamps of that day. With --day-shift N,
   the prediction from N days earlier is assigned instead.
4. Merge hourly IMV, hourly flow, and hourly EDDEV by timestamp (and scenario).
5. Save shifted outputs under data_processed/day_shift_N. Shift 0 keeps the
   original experiments/data_processed paths unchanged.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


# basin_names ={
#     'WatonwanR_Watersheds': 'Watonwan',
#     'LeSueurR_Watersheds': 'LeSueur',
#     'KettleR_Watersheds': 'KettleRiverModels',
#     'BlueEarthR_Watersheds': 'BlueEarth',
#     'LittleFork_Watersheds': 'LittleFork',
#     'SnakeSE_Watersheds': 'SnakeSE',
#     'Zumbro_Watersheds': 'Zumbro',

#     'Cloquet_Watersheds': 'Cloquet',
#     'Sauk_Watersheds': 'Sauk',
#     'SFCrow_Watersheds': 'SFCrow',
#     'TwoRivers_Watersheds': 'TwoRivers',
#     'WildRice_Watersheds': 'WildRiceMarsh'
# }


DEFAULT_WATERSHEDS = [
	"BlueEarth",
	"LeSueur",
	"Watonwan",
	"KettleRiverModels",
	"LittleFork",
	"Zumbro",
	"SnakeSE",
	"Cloquet",
	"Sauk",
	"SFCrow",
	"TwoRivers",
	"WildRiceMarsh"
]
DEFAULT_SPLITS = ["train", "val", "test"]
OBSERVED_IMV_COLS = ["PET", "ET", "SUPY", "WYIE", "SNOW", "TWS", "LZS", "AGW"]


def non_negative_int(value: str) -> int:
	"""Parse a non-negative integer for argparse."""
	try:
		parsed = int(value)
	except ValueError as exc:
		raise argparse.ArgumentTypeError("must be an integer") from exc
	if parsed < 0:
		raise argparse.ArgumentTypeError("must be greater than or equal to 0")
	return parsed


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Combine daily IMV outputs with hourly streamflow and hourly EDDEV "
			"for each watershed and split."
		)
	)
	parser.add_argument(
		"--watersheds",
		nargs="+",
		default=DEFAULT_WATERSHEDS,
		help="Watershed names to process.",
	)
	parser.add_argument(
		"--splits",
		nargs="+",
		default=DEFAULT_SPLITS,
		choices=["train", "val", "test"],
		help="Dataset splits to process.",
	)
	parser.add_argument(
		"--imv-dir",
		type=str,
		default="experiments/data_processed",
		help="Directory containing per-watershed daily IMV outputs.",
	)
	parser.add_argument(
		"--hourly-data-dir",
		type=str,
		default="data_processed",
		help="Root directory containing hourly flow and EDDEV data by watershed.",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default=None,
		help=(
			"Directory to save combined hourly outputs. Defaults to "
			"experiments/data_processed for day shift 0 and "
			"data_processed/day_shift_N for larger shifts."
		),
	)
	parser.add_argument(
		"--day-shift",
		type=non_negative_int,
		default=0,
		help=(
			"Number of days to lag prediction and observed-IMV inputs. For "
			"example, 1 uses the previous day's inputs for the current day. "
			"Defaults to 0 for backward compatibility."
		),
	)
	parser.add_argument(
		"--allow-missing",
		action="store_true",
		help="Continue when any input file is missing for a watershed/split/scenario.",
	)
	return parser.parse_args()


def ensure_dir(path: Path):
	path.mkdir(parents=True, exist_ok=True)


def load_imv_daily(imv_path: Path) -> pd.DataFrame:
	df = pd.read_csv(imv_path)
	if "timestamp" not in df.columns:
		raise KeyError(f"Missing 'timestamp' column in IMV file: {imv_path}")
	if "scenario" not in df.columns:
		raise KeyError(f"Missing 'scenario' column in IMV file: {imv_path}")

	df = df.copy()
	df["timestamp"] = pd.to_datetime(df["timestamp"])
	df["date"] = df["timestamp"].dt.floor("D")
	return df


def resolve_flow_file(hourly_data_dir: Path, watershed: str, scenario: str) -> Path:
	basin_dir = hourly_data_dir / watershed
	preferred = basin_dir / f"{watershed}_{scenario}_flow_hourly.csv"
	fallback = basin_dir / f"{watershed}_{scenario}_flow.csv"

	if preferred.exists():
		return preferred
	if fallback.exists():
		return fallback
	raise FileNotFoundError(
		f"No flow file found for watershed={watershed}, scenario={scenario}. "
		f"Checked: {preferred}, {fallback}"
	)


def resolve_eddev_file(hourly_data_dir: Path, watershed: str, scenario: str) -> Path:
	basin_dir = hourly_data_dir / watershed
	eddev = basin_dir / f"{watershed}_{scenario}_eddev1.csv"
	if not eddev.exists():
		raise FileNotFoundError(
			f"No EDDEV file found for watershed={watershed}, scenario={scenario}. "
			f"Checked: {eddev}"
		)
	return eddev


def load_hourly_flow(path: Path) -> pd.DataFrame:
	df = pd.read_csv(path)
	if "Datetime" not in df.columns:
		raise KeyError(f"Missing 'Datetime' column in flow file: {path}")
	df = df.copy()
	df["Datetime"] = pd.to_datetime(df["Datetime"])
	return df


def load_hourly_eddev(path: Path) -> pd.DataFrame:
	df = pd.read_csv(path)
	if "Datetime" not in df.columns:
		raise KeyError(f"Missing 'Datetime' column in EDDEV file: {path}")
	df = df.copy()
	df["Datetime"] = pd.to_datetime(df["Datetime"])
	return df


def interpolate_daily_imv_to_hourly(
	daily_imv: pd.DataFrame,
	hourly_timestamps: pd.Series,
	scenario: str,
	day_shift: int = 0,
) -> pd.DataFrame:
	"""
	Assign lagged daily IMV records to hourly timestamps.

	A ``day_shift`` of N moves the effective date of every daily prediction
	forward N days. Consequently, hours on date D receive the prediction from
	date D-N. An inner merge intentionally removes hours without an available
	lagged prediction (normally the first N days of a split).
	"""
	if day_shift < 0:
		raise ValueError("day_shift must be greater than or equal to 0")

	# Keep only model prediction columns from IMV outputs; obs_* are redundant
	# with original hourly sources and are intentionally excluded.
	imv_cols = [c for c in daily_imv.columns if c.startswith("pred_")]
	if not imv_cols:
		raise ValueError("No IMV value columns found to interpolate.")

	base = pd.DataFrame({"Datetime": pd.to_datetime(hourly_timestamps)})
	base["date"] = base["Datetime"].dt.floor("D")

	daily_values = daily_imv[["date"] + imv_cols].copy()
	daily_values = daily_values.drop_duplicates(subset=["date"], keep="last")
	daily_values["date"] = daily_values["date"] + pd.to_timedelta(day_shift, unit="D")

	hourly_imv = base.merge(daily_values, on="date", how="inner")
	hourly_imv["scenario"] = scenario

	ordered = ["Datetime", "scenario"] + imv_cols
	return hourly_imv[ordered]


def merge_hourly_sources(
	hourly_flow: pd.DataFrame,
	hourly_eddev: pd.DataFrame,
	hourly_imv: pd.DataFrame,
	scenario: str,
	day_shift: int = 0,
) -> pd.DataFrame:
	# First align flow and EDDEV on timestamp.
	merged = pd.merge(hourly_flow, hourly_eddev, on="Datetime", how="inner")

	# Lag only the observed IMV inputs. The streamflow target and meteorological
	# inputs remain aligned with the current hour. Moving the IMV timestamps
	# forward means an hour on date D receives the value from the same hour on
	# date D-day_shift.
	if day_shift < 0:
		raise ValueError("day_shift must be greater than or equal to 0")
	if day_shift > 0:
		observed_cols = [col for col in OBSERVED_IMV_COLS if col in merged.columns]
		if observed_cols:
			lagged_observed = merged[["Datetime"] + observed_cols].copy()
			lagged_observed["Datetime"] += pd.to_timedelta(day_shift, unit="D")
			merged = merged.drop(columns=observed_cols).merge(
				lagged_observed,
				on="Datetime",
				how="inner",
			)

	merged["scenario"] = scenario

	# Then align with hourly IMV on timestamp + scenario.
	merged = pd.merge(merged, hourly_imv, on=["Datetime", "scenario"], how="inner")

	# Reorder to match the expected schema exactly where columns are available.
	preferred_order = [
		"Datetime",
		"scenario",
		"streamflow",
		"PET",
		"ET",
		"SUPY",
		"WYIE",
		"SNOW",
		"TWS",
		"LZS",
		"AGW",
		"T2",
		"DEWPT",
		"PRECIP",
		"SWDNB",
		"WSPD10",
		"LH",
		"Area_ac_total",
		"pred_AGW",
		"pred_ET",
		"pred_LZS",
		"pred_PET",
		"pred_SNOW",
		"pred_SUPY",
		"pred_TWS",
		"pred_WYIE",
		"pred_streamflow",
	]
	existing_preferred = [col for col in preferred_order if col in merged.columns]
	remaining = [col for col in merged.columns if col not in existing_preferred]
	return merged[existing_preferred + remaining].sort_values("Datetime").reset_index(drop=True)


def process_watershed_split(
	watershed: str,
	split_name: str,
	imv_dir: Path,
	hourly_data_dir: Path,
	allow_missing: bool,
	day_shift: int = 0,
) -> pd.DataFrame:
	imv_path = imv_dir / f"{watershed}_{split_name}_imv_outputs.csv"
	if not imv_path.exists():
		raise FileNotFoundError(f"Missing IMV file: {imv_path}")

	imv_daily = load_imv_daily(imv_path)
	scenario_values = sorted(imv_daily["scenario"].dropna().unique().tolist())

	if not scenario_values:
		raise ValueError(f"No scenario values found in {imv_path}")

	scenario_frames: List[pd.DataFrame] = []
	missing_inputs: List[str] = []

	for scenario in scenario_values:
		scenario_imv = imv_daily[imv_daily["scenario"] == scenario].copy()

		try:
			flow_path = resolve_flow_file(hourly_data_dir, watershed, scenario)
			eddev_path = resolve_eddev_file(hourly_data_dir, watershed, scenario)
			hourly_flow = load_hourly_flow(flow_path)
			hourly_eddev = load_hourly_eddev(eddev_path)
		except Exception as exc:
			if allow_missing:
				missing_inputs.append(str(exc))
				continue
			raise

		hourly_imv = interpolate_daily_imv_to_hourly(
			daily_imv=scenario_imv,
			hourly_timestamps=hourly_flow["Datetime"],
			scenario=scenario,
			day_shift=day_shift,
		)

		combined = merge_hourly_sources(
			hourly_flow=hourly_flow,
			hourly_eddev=hourly_eddev,
			hourly_imv=hourly_imv,
			scenario=scenario,
			day_shift=day_shift,
		)
		scenario_frames.append(combined)

	if missing_inputs:
		print(f"Warnings for {watershed}/{split_name}:")
		for msg in missing_inputs:
			print(f"  - {msg}")

	if not scenario_frames:
		return pd.DataFrame()

	out = pd.concat(scenario_frames, ignore_index=True)
	out = out.sort_values(["scenario", "Datetime"]).reset_index(drop=True)
	return out


def main():
	args = parse_arguments()

	imv_dir = Path(args.imv_dir)
	hourly_data_dir = Path(args.hourly_data_dir)
	default_output_dir = (
		"experiments/data_processed"
		if args.day_shift == 0
		else f"data_processed/day_shift_{args.day_shift}"
	)
	output_dir = Path(args.output_dir or default_output_dir)
	ensure_dir(output_dir)
	print(f"Using day_shift={args.day_shift}; writing outputs to {output_dir}")

	written = 0
	aggregate_written = 0

	for watershed in args.watersheds:
		# Store per-split outputs so we can also build split-combined files.
		split_outputs: Dict[str, pd.DataFrame] = {}
		for split_name in args.splits:
			try:
				combined_df = process_watershed_split(
					watershed=watershed,
					split_name=split_name,
					imv_dir=imv_dir,
					hourly_data_dir=hourly_data_dir,
					allow_missing=args.allow_missing,
					day_shift=args.day_shift,
				)
			except Exception as exc:
				if args.allow_missing:
					print(f"Skipping {watershed}/{split_name}: {exc}")
					continue
				raise

			if combined_df.empty:
				print(f"Skipping {watershed}/{split_name}: no rows after merging.")
				continue

			split_outputs[split_name] = combined_df.copy()

			output_path = output_dir / f"{watershed}_{split_name}_combined_hourly.csv"
			combined_df.to_csv(output_path, index=False)
			written += 1
			print(
				f"Saved {output_path} "
				f"(rows={len(combined_df)}, cols={len(combined_df.columns)})"
			)

		if not split_outputs:
			continue

		# Combine train/val/test together and then save one file per scenario.
		merged_splits: List[pd.DataFrame] = []
		for split_name, split_df in split_outputs.items():
			tmp = split_df.copy()
			tmp["split_name"] = split_name
			merged_splits.append(tmp)

		all_splits_df = pd.concat(merged_splits, ignore_index=True)
		all_splits_df = all_splits_df.sort_values(["Datetime", "split_name"]).reset_index(drop=True)

		for scenario, scenario_df in all_splits_df.groupby("scenario", dropna=False):
			if pd.isna(scenario):
				print(f"Skipping {watershed}: encountered NaN scenario while saving split-combined files.")
				continue

			scenario_out = scenario_df.sort_values("Datetime").reset_index(drop=True)
			output_path = output_dir / f"{watershed}_{scenario}_combined.csv"
			scenario_out.to_csv(output_path, index=False)
			aggregate_written += 1
			print(
				f"Saved {output_path} "
				f"(rows={len(scenario_out)}, cols={len(scenario_out.columns)})"
			)

	print(
		f"Done. Generated {written} split-level combined hourly files and "
		f"{aggregate_written} split-aggregated scenario files in {output_dir}"
	)


if __name__ == "__main__":
	main()
