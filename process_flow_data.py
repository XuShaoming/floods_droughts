import pandas as pd
import numpy as np
import os
from pathlib import Path

def process_flow_data(basin_name, scenario):
    """
    Process flow data by merging hourly flow data with daily output data.
    The daily data will be interpolated to hourly by repeating each day's values 24 times.
    
    Parameters:
    -----------
    basin_name : str
        Name of the basin (e.g., 'KettleRiverModels', 'BlueEarth', 'LeSueur')
    scenario : str
        Climate scenario (e.g., 'hist_scaled', 'RCP4.5', 'RCP8.5')
    
    Returns:
    --------
    pd.DataFrame
        Merged dataframe with hourly flow data and interpolated daily output data
    """
    # Define file paths
    flow_data_dir = Path('/home/kumarv/xu000114/floods_droughts/data/flow_data')
    hourly_file = flow_data_dir / f"{basin_name}_outlet_{scenario}_FLOW.csv"
    daily_file = flow_data_dir / f"{basin_name}_outlet_{scenario}_Daily_outputs.csv"
    
    # Read data
    print(f"Reading hourly flow data from {hourly_file}")
    hourly_df = pd.read_csv(hourly_file)
    print(f"Reading daily output data from {daily_file}")
    daily_df = pd.read_csv(daily_file)
    
    # Convert datetime columns to datetime type
    hourly_df['Datetime'] = pd.to_datetime(hourly_df['Datetime'])
    daily_df['Datetime'] = pd.to_datetime(daily_df['Datetime'])
    
    # Set datetime as index for daily data
    daily_df.set_index('Datetime', inplace=True)
    
    # Create a list to store hourly interpolated daily data
    hourly_daily_data = []
        
    # Process each day in the hourly data
    print("Interpolating daily data to hourly resolution...")
    for date, group in hourly_df.groupby(hourly_df['Datetime'].dt.date):
        # Convert date to datetime
        date = pd.to_datetime(date)
        
        # Check if the date exists in daily data
        if date in daily_df.index:
            # Get daily values for this date
            daily_values = daily_df.loc[date].to_dict()
            
            # Create a dataframe for this day's hourly data
            day_hourly = group.copy()
            
            # Add daily values to each hour
            for col, val in daily_values.items():
                day_hourly[col] = val
                
            hourly_daily_data.append(day_hourly)
        else:
            print(f"Warning: Date {date} not found in daily data")
    
    # Get flow column name (it may vary depending on the basin)
    flow_col = [col for col in hourly_df.columns if col != 'Datetime'][0]

    # Concatenate all daily hourly data
    if hourly_daily_data:
        result_df = pd.concat(hourly_daily_data)
        
        # Reorder columns to have Datetime and flow first, then other variables
        cols = ['Datetime', flow_col] + [col for col in result_df.columns 
                                         if col not in ['Datetime', flow_col]]
        result_df = result_df[cols]

        # Rename flow column to 'streamflow'
        result_df.rename(columns={flow_col: 'streamflow'}, inplace=True)

        return result_df
    else:
        print("No matching dates found between hourly and daily data")
        return None

if __name__ == "__main__":
    # Create processed directory if it doesn't exist
    processed_dir = Path('/home/kumarv/xu000114/floods_droughts/data_processed')
    processed_dir.mkdir(exist_ok=True)
    
    # Process data for each basin and scenario
    # basins = ['KettleRiverModels', 'BlueEarth', 'LeSueur', 'Watonwan']
    # basins = ['KettleRiverModels']
    basins = ['BlueEarth', 'LeSueur', 'Watonwan']
    scenarios = ['hist_scaled', 'RCP4.5', 'RCP8.5']
    
    # Dictionary to store all processed data    
    for basin in basins:
        for scenario in scenarios:
            print(f"\nProcessing {basin} - {scenario}")
            try:
                processed_df = process_flow_data(basin, scenario)
                if processed_df is not None:
                    output_file = processed_dir / basin / f"{basin}_{scenario}_flow.csv"
                    output_file.parent.mkdir(exist_ok=True)
                    processed_df.to_csv(output_file, index=False)
                    print(f"Saved processed data to {output_file}")
            except Exception as e:
                print(f"Error processing {basin} - {scenario}: {e}")