#!/usr/bin/env python3
"""Analyze and visualize USGS observed streamflow data."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd


@dataclass
class FlowData:
	label: str
	df: pd.DataFrame
	value_col: str
	file_path: str


def _find_value_column(df: pd.DataFrame) -> str:
	candidates = [c for c in df.columns if c.lower() != "datetime"]
	if not candidates:
		raise ValueError("No flow column found (expected a non-datetime column).")
	if len(candidates) > 1:
		numeric_cols = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
		if numeric_cols:
			return numeric_cols[0]
	return candidates[0]


def read_flow_csv(file_path: str, label: str) -> FlowData:
	df = pd.read_csv(file_path)
	if "datetime" not in df.columns:
		raise ValueError(f"Missing 'datetime' column in {file_path}")
	df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
	df = df.dropna(subset=["datetime"]).sort_values("datetime")
	value_col = _find_value_column(df)
	df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
	return FlowData(label=label, df=df, value_col=value_col, file_path=file_path)


def basic_stats(series: pd.Series) -> pd.Series:
	return pd.Series(
		{
			"count": series.count(),
			"min": series.min(),
			"max": series.max(),
			"mean": series.mean(),
			"median": series.median(),
			"std": series.std(),
			"cv": series.std() / series.mean() if series.mean() != 0 else np.nan,
			"p01": series.quantile(0.01),
			"p05": series.quantile(0.05),
			"p10": series.quantile(0.10),
			"p25": series.quantile(0.25),
			"p75": series.quantile(0.75),
			"p90": series.quantile(0.90),
			"p95": series.quantile(0.95),
			"p99": series.quantile(0.99),
		}
	)


def flow_duration_curve(series: pd.Series) -> pd.DataFrame:
	sorted_vals = series.dropna().sort_values(ascending=False)
	n = len(sorted_vals)
	exceedance = (np.arange(1, n + 1) / (n + 1)) * 100.0
	return pd.DataFrame({"exceedance_percent": exceedance, "flow": sorted_vals.values})


def annual_metrics(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
	annual = df.set_index("datetime")[value_col].resample("YE").agg([
		"mean",
		"max",
		"min",
	])
	annual.index = annual.index.year
	annual = annual.rename_axis("year").reset_index()
	return annual


def rolling_mean(df: pd.DataFrame, value_col: str, window_days: int) -> pd.DataFrame:
	series = df.set_index("datetime")[value_col].resample("D").mean()
	rolling_min = series.rolling(window=window_days, min_periods=window_days).mean()
	out = rolling_min.to_frame(name=f"{window_days}d_mean")
	return out.reset_index()


def monthly_climatology(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
	series = df.set_index("datetime")[value_col].resample("D").mean()
	out = series.groupby(series.index.month).agg(["mean", "median", "max", "min"])
	out.index.name = "month"
	return out.reset_index()


def save_table(df: pd.DataFrame, path: str) -> None:
	df.to_csv(path, index=False)


def plot_time_series(flow: FlowData, output_path: str) -> None:
	df = flow.df.copy()
	df["datetime"] = df["datetime"].dt.tz_convert(None)
	missing_mask = df[flow.value_col].isna() | (df[flow.value_col] < 0)
	valid_df = df[~missing_mask]
	missing_df = df[missing_mask]
	total_points = len(df)
	missing_points = int(missing_mask.sum())
	plot_values = df[flow.value_col].copy()
	plot_values[missing_mask] = -1.0
	plt.figure(figsize=(12, 4))
	if total_points > 1:
		x = mdates.date2num(df["datetime"]).astype(float)
		y = plot_values.values
		points = np.column_stack([x, y]).reshape(-1, 1, 2)
		segments = np.concatenate([points[:-1], points[1:]], axis=1)
		seg_missing = missing_mask.values[:-1] | missing_mask.values[1:]
		colors = np.where(seg_missing, "red", "tab:blue")
		lc = LineCollection(segments, colors=colors, linewidths=0.8)
		plt.gca().add_collection(lc)
		plt.gca().set_xlim(x.min(), x.max())
		plt.gca().xaxis_date()
		y_min = np.nanmin(y)
		y_max = np.nanmax(y)
		if np.isfinite(y_min) and np.isfinite(y_max):
			plt.gca().set_ylim(y_min, y_max)
	elif total_points == 1:
		plt.plot(df["datetime"], plot_values, color="tab:blue", linewidth=0.8)
	plt.title(
		f"{flow.label} - Time Series | total={total_points}, missing={missing_points}"
	)
	plt.xlabel("Date")
	plt.ylabel("Streamflow")
	plt.tight_layout()
	plt.savefig(output_path, dpi=200)
	plt.close()


def plot_monthly_climatology(clim: pd.DataFrame, label: str, output_path: str) -> None:
	plt.figure(figsize=(10, 4))
	plt.plot(clim["month"], clim["mean"], marker="o", label="Mean")
	plt.plot(clim["month"], clim["median"], marker="o", label="Median")
	plt.fill_between(clim["month"], clim["min"], clim["max"], alpha=0.2, label="Min-Max")
	plt.title(f"{label} - Monthly Climatology")
	plt.xlabel("Month")
	plt.ylabel("Streamflow")
	plt.xticks(range(1, 13))
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_path, dpi=200)
	plt.close()


def plot_flow_duration(fdc: pd.DataFrame, label: str, output_path: str) -> None:
	plt.figure(figsize=(6, 4))
	plt.semilogy(fdc["exceedance_percent"], fdc["flow"], linewidth=1.0)
	plt.gca().invert_xaxis()
	plt.title(f"{label} - Flow Duration Curve")
	plt.xlabel("Exceedance Probability (%)")
	plt.ylabel("Streamflow")
	plt.tight_layout()
	plt.savefig(output_path, dpi=200)
	plt.close()


def plot_annual_max(annual: pd.DataFrame, label: str, output_path: str) -> None:
	plt.figure(figsize=(10, 4))
	plt.plot(annual["year"], annual["max"], marker="o", linewidth=0.8)
	plt.title(f"{label} - Annual Max Flow")
	plt.xlabel("Year")
	plt.ylabel("Streamflow")
	plt.tight_layout()
	plt.savefig(output_path, dpi=200)
	plt.close()


def analyze_flow(flow: FlowData, output_dir: str) -> None:
	missing_mask = flow.df[flow.value_col].isna() | (flow.df[flow.value_col] < 0)
	df_valid = flow.df[~missing_mask].copy()
	series = df_valid[flow.value_col]
	stats = basic_stats(series)
	stats_df = stats.to_frame(name="value").reset_index().rename(columns={"index": "metric"})
	save_table(stats_df, os.path.join(output_dir, f"{flow.label}_stats.csv"))

	fdc = flow_duration_curve(series)
	save_table(fdc, os.path.join(output_dir, f"{flow.label}_fdc.csv"))

	annual = annual_metrics(df_valid, flow.value_col)
	save_table(annual, os.path.join(output_dir, f"{flow.label}_annual_metrics.csv"))

	lowflow7 = rolling_mean(df_valid, flow.value_col, window_days=7)
	save_table(lowflow7, os.path.join(output_dir, f"{flow.label}_7d_mean.csv"))

	climatology = monthly_climatology(df_valid, flow.value_col)
	save_table(climatology, os.path.join(output_dir, f"{flow.label}_monthly_climatology.csv"))

	plot_time_series(flow, os.path.join(output_dir, f"{flow.label}_timeseries.png"))
	plot_flow_duration(fdc, flow.label, os.path.join(output_dir, f"{flow.label}_fdc.png"))
	plot_annual_max(annual, flow.label, os.path.join(output_dir, f"{flow.label}_annual_max.png"))
	plot_monthly_climatology(
		climatology,
		flow.label,
		os.path.join(output_dir, f"{flow.label}_monthly_climatology.png"),
	)


def resolve_files(base_dir: str, watershed: str, gauge: str) -> Tuple[str, str]:
	daily = os.path.join(
		base_dir, watershed, f"{watershed}_obs_daily_{gauge}.csv"
	)
	inst = os.path.join(
		base_dir, watershed, f"{watershed}_obs_inst_{gauge}.csv"
	)
	return daily, inst


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Analyze daily and instantaneous USGS observed flow data."
	)
	parser.add_argument("--watershed", required=True, help="Watershed folder name")
	parser.add_argument("--gauge", required=True, help="USGS gauge number")
	parser.add_argument(
		"--base-dir",
		default="/projects/standard/kumarv/xu000114/floods_droughts/data/flow_data",
		help="Base directory for flow data",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	daily_path, inst_path = resolve_files(args.base_dir, args.watershed, args.gauge)

	if not os.path.exists(daily_path):
		raise FileNotFoundError(f"Daily file not found: {daily_path}")
	if not os.path.exists(inst_path):
		raise FileNotFoundError(f"Instantaneous file not found: {inst_path}")

	daily_flow = read_flow_csv(daily_path, f"{args.watershed}_daily")
	inst_flow = read_flow_csv(inst_path, f"{args.watershed}_inst")

	output_dir = os.path.join(os.path.dirname(daily_path), "analysis")
	os.makedirs(output_dir, exist_ok=True)

	analyze_flow(daily_flow, output_dir)
	analyze_flow(inst_flow, output_dir)


if __name__ == "__main__":
	main()
