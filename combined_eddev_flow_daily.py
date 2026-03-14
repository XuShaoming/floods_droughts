#!/usr/bin/env python3
"""
Convert combined EDDEV + flow data from hourly to daily resolution.

This script reads a combined file produced by combine_eddev_flow.py and
aggregates it to daily values using mean-by-day for numeric columns.

Usage:
	python data_processed/combined_eddev_flow_daily.py --basin KettleRiverModels --scenario hist_scaled
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_arguments():
	"""Parse command line arguments."""
	parser = argparse.ArgumentParser(
		description="Convert combined hourly EDDEV + flow data to daily data"
	)
	parser.add_argument(
		"--basin",
		type=str,
		required=True,
		help="Basin name (e.g., KettleRiverModels, BlueEarth, LeSueur)",
	)
	parser.add_argument(
		"--scenario",
		type=str,
		required=True,
		help="Climate scenario (e.g., hist_scaled, RCP4.5, RCP8.5)",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default="data_processed",
		help="Directory containing/saving processed data (default: data_processed)",
	)
	parser.add_argument(
		"--resolution",
		type=str,
		choices=["hourly", "daily"],
		default="hourly",
		help=(
			"Resolution tag of combined input file from combine_eddev_flow.py "
			"(default: hourly)"
		),
	)
	return parser.parse_args()


def convert_to_daily(input_file: Path) -> pd.DataFrame:
	"""
	Load combined data from a CSV file and aggregate it to daily means.

	Parameters:
		input_file (Path): Path to the input CSV file containing combined data.

	Returns:
		pd.DataFrame: DataFrame aggregated to daily resolution.
	"""
	# Inform the user which file is being read
	print(f"Reading combined data from {input_file}")
	df = pd.read_csv(input_file)

	# Ensure the required 'Datetime' column exists
	if "Datetime" not in df.columns:
		raise KeyError("Input file must contain a 'Datetime' column.")

	# Convert 'Datetime' column to datetime objects and set as index
	df["Datetime"] = pd.to_datetime(df["Datetime"])
	df = df.sort_values("Datetime").set_index("Datetime")

	# Identify numeric and non-numeric columns
	numeric_cols = df.select_dtypes(include="number").columns.tolist()
	non_numeric_cols = [col for col in df.columns if col not in numeric_cols]

	# Raise an error if there are no numeric columns to aggregate
	if not numeric_cols:
		raise ValueError("No numeric columns found to average by day.")

	# Aggregate numeric columns by daily mean
	daily_numeric = df[numeric_cols].resample("D").mean()

	# For non-numeric columns, take the first value for each day (e.g., metadata)
	if non_numeric_cols:
		daily_non_numeric = df[non_numeric_cols].resample("D").first()
		# Combine numeric and non-numeric daily data
		daily_df = pd.concat([daily_numeric, daily_non_numeric], axis=1)
	else:
		# If no non-numeric columns, use only numeric daily data
		daily_df = daily_numeric

	# Reset index to turn 'Datetime' back into a column
	daily_df = daily_df.reset_index()
	return daily_df


if __name__ == "__main__":
	args = parse_arguments()

	processed_dir = Path(args.output_dir)
	processed_dir.mkdir(exist_ok=True)

	# res_tag = "" if args.resolution == "daily" else "_hourly"
	res_tag = ""

	input_file = processed_dir / f"{args.basin}_{args.scenario}_combined{res_tag}.csv"
	output_file = processed_dir / f"{args.basin}_{args.scenario}_combined_daily.csv"

	if not input_file.exists():
		raise FileNotFoundError(f"Input file not found: {input_file}")

	daily_df = convert_to_daily(input_file)

	print(f"Saving daily data to {output_file}")
	daily_df.to_csv(output_file, index=False)
	print(f"Done. Wrote {len(daily_df)} daily rows.")
