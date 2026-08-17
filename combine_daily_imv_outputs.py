#!/usr/bin/env python3
"""
Combine reconstructed IMV outputs across experiments into per-watershed files.

This script reads reconstructed files produced by inference_global.py with names
like:

	{watershed}_{split_name}_reconstructed_{method}.{file_format}

It then merges all configured IMV experiments into a single file per watershed
and split, and saves outputs as:

	experiments/data_processed/{watershed}_{split_name}_imv_outputs.csv
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

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

DEFAULT_SOURCE_EXPERIMENTS = [
	"daily_global_TWS",
	"daily_global_LZS",
	"daily_global_AGW",
	"daily_global_SNOW",
	"daily_global_SUPY",
	"daily_global_WYIE",
	"daily_global_ET",
	"daily_global_PET",
	"daily_global_streamflow",
]


def ensure_dir(path: str):
	os.makedirs(path, exist_ok=True)


def load_dataframe(path: str, file_format: str) -> pd.DataFrame:
	if file_format == "csv":
		return pd.read_csv(path)
	if file_format == "parquet":
		return pd.read_parquet(path)
	raise ValueError(f"Unsupported file format '{file_format}'")


def save_dataframe(df: pd.DataFrame, path: str, file_format: str):
	ensure_dir(os.path.dirname(path))
	if file_format == "csv":
		df.to_csv(path, index=False)
	elif file_format == "parquet":
		df.to_parquet(path, index=False)
	else:
		raise ValueError(f"Unsupported file format '{file_format}'")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Combine IMV inference outputs per watershed and split."
	)
	parser.add_argument(
		"--experiments-dir",
		type=str,
		default="experiments",
		help="Root experiments directory containing inference output folders.",
	)
	parser.add_argument(
		"--source-experiments",
		nargs="+",
		default=DEFAULT_SOURCE_EXPERIMENTS,
		help="Experiment folders under --experiments-dir to merge.",
	)
	parser.add_argument(
		"--results-subdir",
		type=str,
		default="test_results",
		help="Subdirectory inside each source experiment where reconstructed files exist.",
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
		"--method",
		type=str,
		default="latest",
		help="Reconstruction method suffix used by inference outputs.",
	)
	parser.add_argument(
		"--input-format",
		type=str,
		default="csv",
		choices=["csv", "parquet"],
		help="File format of inference outputs.",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default=None,
		help="Output directory for combined files. Defaults to experiments/data_processed.",
	)
	parser.add_argument(
		"--output-format",
		type=str,
		default="csv",
		choices=["csv", "parquet"],
		help="Output file format for combined IMV files.",
	)
	parser.add_argument(
		"--allow-missing",
		action="store_true",
		help="If set, allow missing source files and continue with available inputs.",
	)
	return parser.parse_args()


def build_reconstructed_path(
	experiments_dir: str,
	source_experiment: str,
	results_subdir: str,
	watershed: str,
	split_name: str,
	method: str,
	file_format: str,
) -> str:
	recon_filename = f"{watershed}_{split_name}_reconstructed_{method}.{file_format}"
	return os.path.join(experiments_dir, source_experiment, results_subdir, recon_filename)


def normalize_reconstructed_df(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
	if "timestamp" not in df.columns:
		raise ValueError(f"Missing required 'timestamp' column in source '{source_label}'.")

	out = df.copy()
	out["timestamp"] = pd.to_datetime(out["timestamp"])

	if "scenario" not in out.columns:
		out["scenario"] = pd.NA

	keep_cols = [
		col
		for col in out.columns
		if col == "timestamp" or col == "scenario" or col.startswith("pred_") or col.startswith("obs_")
	]

	if len(keep_cols) <= 2:
		raise ValueError(
			f"No prediction/observation columns found in source '{source_label}'. "
			f"Expected columns like pred_*/obs_*."
		)

	return out[keep_cols]


def merge_sources(dataframes: List[pd.DataFrame], labels: List[str]) -> pd.DataFrame:
	if not dataframes:
		return pd.DataFrame()

	merged = dataframes[0]
	existing_value_cols = {c for c in merged.columns if c not in {"timestamp", "scenario"}}

	for idx in range(1, len(dataframes)):
		current = dataframes[idx]
		current_value_cols = {c for c in current.columns if c not in {"timestamp", "scenario"}}
		overlap = sorted(existing_value_cols.intersection(current_value_cols))
		if overlap:
			raise ValueError(
				"Duplicate IMV columns found while merging "
				f"'{labels[idx - 1]}' and '{labels[idx]}': {overlap}"
			)

		merged = merged.merge(current, on=["timestamp", "scenario"], how="outer")
		existing_value_cols.update(current_value_cols)

	sort_cols = ["timestamp"]
	if "scenario" in merged.columns:
		sort_cols.append("scenario")
	merged = merged.sort_values(sort_cols).reset_index(drop=True)

	ordered_cols = ["timestamp"]
	if "scenario" in merged.columns:
		ordered_cols.append("scenario")
	ordered_cols.extend(
		sorted(c for c in merged.columns if c.startswith("pred_"))
	)
	ordered_cols.extend(
		sorted(c for c in merged.columns if c.startswith("obs_"))
	)
	return merged[ordered_cols]


def main():
	args = parse_args()

	experiments_dir = args.experiments_dir
	output_dir = args.output_dir or os.path.join(experiments_dir, "data_processed")
	ensure_dir(output_dir)

	total_outputs = 0

	for watershed in args.watersheds:
		for split_name in args.splits:
			source_dfs: List[pd.DataFrame] = []
			source_labels: List[str] = []
			missing_paths: List[str] = []

			for source_experiment in args.source_experiments:
				recon_path = build_reconstructed_path(
					experiments_dir=experiments_dir,
					source_experiment=source_experiment,
					results_subdir=args.results_subdir,
					watershed=watershed,
					split_name=split_name,
					method=args.method,
					file_format=args.input_format,
				)

				if not os.path.exists(recon_path):
					missing_paths.append(recon_path)
					continue

				recon_df = load_dataframe(recon_path, args.input_format)
				recon_df = normalize_reconstructed_df(recon_df, source_experiment)
				source_dfs.append(recon_df)
				source_labels.append(source_experiment)

			if missing_paths and not args.allow_missing:
				raise FileNotFoundError(
					"Missing reconstructed files for "
					f"watershed='{watershed}', split='{split_name}':\n"
					+ "\n".join(missing_paths)
					+ "\nUse --allow-missing to continue with available sources."
				)

			if not source_dfs:
				print(
					f"Skipping {watershed}/{split_name}: no reconstructed files found in configured sources."
				)
				continue

			combined_df = merge_sources(source_dfs, source_labels)

			out_ext = args.output_format
			out_name = f"{watershed}_{split_name}_imv_outputs.{out_ext}"
			out_path = os.path.join(output_dir, out_name)
			save_dataframe(combined_df, out_path, args.output_format)

			total_outputs += 1
			print(
				f"Saved {out_path} (rows={len(combined_df)}, cols={len(combined_df.columns)}, "
				f"sources={len(source_dfs)})"
			)

	print(f"Done. Generated {total_outputs} combined IMV output files in {output_dir}")


if __name__ == "__main__":
	main()
