#!/usr/bin/env python3
"""
Combine EDDEV and Flow Data

This script combines processed flow data and EDDEV meteorological data into a single CSV file
by aligning their datetime values. It merges data from:
1. Flow data processed by process_flow_data.py (streamflow and related metrics)
2. EDDEV1 data processed by process_eddev_data.py (meteorological variables)

The script searches for matching files in the processed directory based on basin and scenario,
then performs an inner join on the datetime column to create aligned datasets.

Usage:
    python combine_eddev_flow.py --basin "KettleRiverModels" --scenario "hist_scaled"
    python combine_eddev_flow.py --basin "KettleRiverModels" --scenario "RCP4.5"
    python combine_eddev_flow.py --basin "KettleRiverModels" --scenario "RCP8.5"
"""

import pandas as pd
import os
import argparse
from pathlib import Path
import glob

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Combine EDDEV and Flow data by datetime')
    parser.add_argument('--basin', type=str, required=True, 
                        help='Basin name (e.g., KettleRiverModels, BlueEarth, LeSueur)')
    parser.add_argument('--scenario', type=str, required=True, 
                        help='Climate scenario. For flow: hist_scaled, RCP4.5, RCP8.5. For EDDEV: Historical, RCP4.5, RCP8.5')
    parser.add_argument('--output_dir', type=str, default='processed',
                        help='Directory to save combined data (default: processed)')
    return parser.parse_args()

def find_matching_files(basin, scenario, processed_dir):
    """
    Find matching flow and EDDEV files for the specified basin and scenario.
    
    Parameters:
    -----------
    basin : str
        Basin name (e.g., 'KettleRiverModels')
    scenario : str
        Climate scenario (e.g., 'hist_scaled' or 'RCP4.5')
    processed_dir : Path
        Directory containing processed data files
    
    Returns:
    --------
    tuple
        (flow_file, eddev_file) paths
    """
    # Map flow scenario names to EDDEV scenario names
    scenario_map = {
        'hist_scaled': 'Historical',
        'RCP4.5': 'RCP4.5',
        'RCP8.5': 'RCP8.5'
    }

    basin_map = {
        'KettleRiverModels': 'KettleR_Watersheds'
    }
    
    # For EDDEV data, we need to map the scenario name
    eddev_scenario = scenario_map.get(scenario, scenario)
    
    # Find flow data file
    flow_file_pattern = str(processed_dir / f"{basin}_{scenario}_flow.csv")
    flow_files = glob.glob(flow_file_pattern)
    
    # Find EDDEV data file
    eddev_pattern = str(processed_dir / f"{basin_map.get(basin, basin)}_{eddev_scenario}_eddev1.csv")
    eddev_files = glob.glob(eddev_pattern)
    
    # If no exact match for EDDEV, try alternative basin name formatting
    if not eddev_files:
        raise ValueError(f"No EDDEV data file found for basin {basin} and scenario {eddev_scenario}.")
    if not flow_files:
        raise FileNotFoundError(f"No flow data file found matching pattern: {flow_file_pattern}")
    
    return flow_files[0], eddev_files[0]

def combine_data(flow_file, eddev_file):
    """
    Combine flow and EDDEV data by datetime.
    
    Parameters:
    -----------
    flow_file : str
        Path to flow data CSV file
    eddev_file : str
        Path to EDDEV data CSV file
    
    Returns:
    --------
    pd.DataFrame
        Combined dataframe with data from both sources aligned by datetime
    """
    print(f"Reading flow data from {flow_file}")
    flow_df = pd.read_csv(flow_file)
    
    print(f"Reading EDDEV data from {eddev_file}")
    eddev_df = pd.read_csv(eddev_file)
    
    # Report initial dataset sizes
    print(f"Initial dataset sizes:")
    print(f"  Flow data: {len(flow_df)} rows")
    print(f"  EDDEV data: {len(eddev_df)} rows")
    print(f"  Difference: {abs(len(flow_df) - len(eddev_df))} rows")
    
    # Convert datetime columns to datetime type
    flow_df['Datetime'] = pd.to_datetime(flow_df['Datetime'])
    eddev_df['Datetime'] = pd.to_datetime(eddev_df['Datetime'])
    
    # Check time range of both datasets
    flow_start = flow_df['Datetime'].min()
    flow_end = flow_df['Datetime'].max()
    eddev_start = eddev_df['Datetime'].min()
    eddev_end = eddev_df['Datetime'].max()
    
    print(f"\nDataset time ranges:")
    print(f"  Flow data: {flow_start} to {flow_end}")
    print(f"  EDDEV data: {eddev_start} to {eddev_end}")
    
    # Check time frequency of both datasets
    flow_time_diff = flow_df['Datetime'].diff().mode().iloc[0]
    eddev_time_diff = eddev_df['Datetime'].diff().mode().iloc[0]
    
    print(f"\nTime frequency:")
    print(f"  Flow data: {flow_time_diff}")
    print(f"  EDDEV data: {eddev_time_diff}")
    
    # Find missing timestamps
    flow_dates_set = set(flow_df['Datetime'])
    eddev_dates_set = set(eddev_df['Datetime'])
    
    missing_in_eddev = flow_dates_set - eddev_dates_set
    missing_in_flow = eddev_dates_set - flow_dates_set
    
    if missing_in_eddev:
        print(f"\nFound {len(missing_in_eddev)} timestamps in flow data that are missing in EDDEV data")
        missing_in_eddev_sorted = sorted(list(missing_in_eddev))
        print(f"  First 5 missing timestamps: {missing_in_eddev_sorted[:5]}")
        print(f"  Last 5 missing timestamps: {missing_in_eddev_sorted[-5:]}")
    
    if missing_in_flow:
        print(f"\nFound {len(missing_in_flow)} timestamps in EDDEV data that are missing in flow data")
        missing_in_flow_sorted = sorted(list(missing_in_flow))
        print(f"  First 5 missing timestamps: {missing_in_flow_sorted[:5]}")
        print(f"  Last 5 missing timestamps: {missing_in_flow_sorted[-5:]}")
    
    # Perform inner join on datetime to align datasets
    print("\nMerging datasets on Datetime using inner join...")
    combined_df = pd.merge(flow_df, eddev_df, on='Datetime', how='inner')
    
    print(f"Combined data has {len(combined_df)} rows (kept timestamps present in both datasets)")
    print(f"Dropped {len(flow_df) - len(combined_df)} rows from flow data")
    print(f"Dropped {len(eddev_df) - len(combined_df)} rows from EDDEV data")
    
    # Check for overlapping column names (other than Datetime)
    flow_columns = set(flow_df.columns)
    eddev_columns = set(eddev_df.columns)
    overlapping = flow_columns.intersection(eddev_columns) - {'Datetime'}
    
    if overlapping:
        print(f"\nWarning: Overlapping column names found: {overlapping}")
        # Rename overlapping columns in the combined dataframe
        for col in overlapping:
            if col in combined_df.columns:
                combined_df.rename(columns={f"{col}_x": f"{col}_flow", f"{col}_y": f"{col}_eddev"}, inplace=True)
    
    return combined_df

if __name__ == "__main__":
    """Main function to combine EDDEV and flow data."""
    args = parse_arguments()
    
    # Ensure output directory exists
    processed_dir = Path(args.output_dir)
    processed_dir.mkdir(exist_ok=True)
    
    try:
        # Find matching files
        flow_file, eddev_file = find_matching_files(args.basin, args.scenario, processed_dir)
        
        # Combine data
        combined_df = combine_data(flow_file, eddev_file)
        
        # Create output filename
        scenario_suffix = args.scenario
        output_file = processed_dir / f"{args.basin}_{scenario_suffix}_combined.csv"
        
        # Save combined data
        print(f"Saving combined data to {output_file}")
        combined_df.to_csv(output_file, index=False)
        print(f"Successfully combined data and saved to {output_file}")
        
        # Print sample of combined data
        print("\nSample of combined data:")
        print(combined_df.head().to_string())
        
    except Exception as e:
        print(f"Error combining data: {e}")