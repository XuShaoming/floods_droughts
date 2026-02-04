import pandas as pd
import numpy as np
from pathlib import Path

def process_flow_data(basin_name, scenario):
    """
    Process flow data by merging hourly flow data with output data.
    
    Parameters:
    -----------
    basin_name : str
        Name of the basin (e.g., 'KettleRiverModels', 'BlueEarth', 'LeSueur')
    scenario : str
        Climate scenario (e.g., 'hist_scaled', 'RCP4.5', 'RCP8.5')
    
    Returns:
    --------
    pd.DataFrame
        Merged dataframe with hourly flow data and hourly output data
    """
    # Define file paths
    flow_data_dir = Path('/projects/standard/kumarv/xu000114/floods_droughts/data/flow_data')
    hourly_flow_file = flow_data_dir / f"{basin_name}_outlet_{scenario}_FLOW.csv"
    hourly_output_file = flow_data_dir / f"{basin_name}_outlet_{scenario}_Hourly_outputs.csv"
    
    # Read data
    print(f"Reading hourly flow data from {hourly_flow_file}")
    hourly_flow_df = pd.read_csv(hourly_flow_file)
    print(f"Reading hourly output data from {hourly_output_file}")
    hourly_output_df = pd.read_csv(hourly_output_file)
    
    # Convert datetime columns to datetime type
    hourly_flow_df['Datetime'] = pd.to_datetime(hourly_flow_df['Datetime'], errors='coerce')
    hourly_output_df['Datetime'] = pd.to_datetime(hourly_output_df['Datetime'], errors='coerce')

    # Drop rows with invalid datetimes (prevents silent misalignment)
    hourly_flow_df = hourly_flow_df.dropna(subset=['Datetime'])
    hourly_output_df = hourly_output_df.dropna(subset=['Datetime'])

    # Get flow column name (it may vary depending on the basin)
    flow_cols = [col for col in hourly_flow_df.columns if col != 'Datetime']
    if len(flow_cols) != 1:
        raise ValueError(
            f"Expected exactly 1 flow column besides 'Datetime', found {len(flow_cols)}: {flow_cols}"
        )
    flow_col = flow_cols[0]

    # Align on Datetime, then concat along columns
    # Using index-aligned concat avoids duplicated 'Datetime' columns and guards against row-order mismatch.
    hourly_flow_df = hourly_flow_df.sort_values('Datetime').set_index('Datetime')
    hourly_output_df = hourly_output_df.sort_values('Datetime').set_index('Datetime')

    if len(hourly_flow_df) != len(hourly_output_df):
        print(
            f"Warning: row count differs for {basin_name}-{scenario}: "
            f"flow={len(hourly_flow_df)}, outputs={len(hourly_output_df)}. "
            "Using inner join on Datetime."
        )

    result_df = pd.concat(
        [hourly_flow_df[[flow_col]], hourly_output_df],
        axis=1,
        join='inner',
        copy=False,
    ).reset_index()

    # Rename flow column to 'streamflow'
    result_df.rename(columns={flow_col: 'streamflow'}, inplace=True)

    # Reorder columns to have Datetime and streamflow first, then other variables
    cols = ['Datetime', 'streamflow'] + [
        col for col in result_df.columns if col not in ['Datetime', 'streamflow']
    ]
    result_df = result_df[cols]

    return result_df

if __name__ == "__main__":
    # Create processed directory if it doesn't exist
    processed_dir = Path('/projects/standard/kumarv/xu000114/floods_droughts/data_processed')
    processed_dir.mkdir(exist_ok=True)
    
    # Process data for each basin and scenario
    # basins = ['KettleRiverModels', 'BlueEarth', 'LeSueur', 'Watonwan']
    basins = ['KettleRiverModels']
    # basins = ['BlueEarth', 'LeSueur', 'Watonwan']
    scenarios = ['hist_scaled', 'RCP4.5', 'RCP8.5']
    
    # Dictionary to store all processed data    
    for basin in basins:
        for scenario in scenarios:
            print(f"\nProcessing {basin} - {scenario}")
            try:
                processed_df = process_flow_data(basin, scenario)
                if processed_df is not None:
                    output_file = processed_dir / basin / f"{basin}_{scenario}_flow_hourly.csv"
                    output_file.parent.mkdir(exist_ok=True)
                    processed_df.to_csv(output_file, index=False)
                    print(f"Saved processed data to {output_file}")
            except Exception as e:
                print(f"Error processing {basin} - {scenario}: {e}")