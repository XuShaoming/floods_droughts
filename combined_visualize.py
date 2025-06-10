import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import argparse
from datetime import datetime

def visualize_streamflow_and_precip(csv_file, start_date=None, end_date=None, output_path=None):
    """
    Visualize streamflow and precipitation data from a CSV file.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing the data
    start_date : str, optional
        Start date for visualization in format 'YYYY-MM-DD'
    end_date : str, optional
        End date for visualization in format 'YYYY-MM-DD'
    output_path : str, optional
        Path to save the output figure
    """
    # Create imgs directory if it doesn't exist
    if not os.path.exists('imgs'):
        os.makedirs('imgs')
    
    # Read the CSV file
    print(f"Reading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Convert datetime column to datetime format
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # Set Datetime as index
    df.set_index('Datetime', inplace=True)
    
    # Filter data based on start and end dates if provided
    if start_date:
        start_date = pd.to_datetime(start_date)
        df = df[df.index >= start_date]
    
    if end_date:
        end_date = pd.to_datetime(end_date)
        df = df[df.index <= end_date]
    
    if df.empty:
        print("No data found for the specified date range")
        return
    
    # Create a figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot streamflow on the first y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Streamflow', color=color)
    ax1.plot(df.index, df['streamflow'], color=color, label='Streamflow')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Format x-axis based on the time span
    time_span = df.index[-1] - df.index[0]
    if time_span.days <= 7:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d-%H'))
    else:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))

    plt.xticks(rotation=45)
    
    # Create a second y-axis for precipitation
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Precipitation', color=color)
    ax2.plot(df.index, df['PRECIP'], color=color, label='Precipitation')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add a title
    time_range = f"from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"
    plt.title(f'Streamflow and Precipitation {time_range}')
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if output path is provided
    if not output_path:
        start_str = df.index[0].strftime('%Y%m%d')
        end_str = df.index[-1].strftime('%Y%m%d')
        output_path = f'imgs/streamflow_precip_{start_str}_{end_str}.png'
    
    plt.savefig(output_path, dpi=300)
    print(f"Figure saved to {output_path}")
    
    # Also display the figure
    plt.show()

def visualize_streamflow_precip_and_feature(csv_file, third_feature, start_date=None, end_date=None, output_path=None):
    """
    Visualize streamflow, precipitation, and a third feature from a CSV file.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing the data
    third_feature : str
        Name of the third feature to visualize (e.g., 'ET', 'PET', 'SNOW')
    start_date : str, optional
        Start date for visualization in format 'YYYY-MM-DD'
    end_date : str, optional
        End date for visualization in format 'YYYY-MM-DD'
    output_path : str, optional
        Path to save the output figure
    """
    # Create imgs directory if it doesn't exist
    if not os.path.exists('imgs'):
        os.makedirs('imgs')
    
    # Read the CSV file
    print(f"Reading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Check if the third feature exists in the dataframe
    if third_feature not in df.columns:
        print(f"Error: {third_feature} not found in the csv file. Available columns: {', '.join(df.columns)}")
        return
    
    # Convert datetime column to datetime format
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # Set Datetime as index
    df.set_index('Datetime', inplace=True)
    
    # Filter data based on start and end dates if provided
    if start_date:
        start_date = pd.to_datetime(start_date)
        df = df[df.index >= start_date]
    
    if end_date:
        end_date = pd.to_datetime(end_date)
        df = df[df.index <= end_date]
    
    if df.empty:
        print("No data found for the specified date range")
        return
    
    # Create a figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot streamflow on the first y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Streamflow', color=color)
    line1 = ax1.plot(df.index, df['streamflow'], color=color, label='Streamflow')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Format x-axis based on the time span
    time_span = df.index[-1] - df.index[0]
    if time_span.days <= 7:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d-%H'))
    else:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))

    plt.xticks(rotation=45)
    
    # Create a second y-axis for precipitation
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Precipitation', color=color)
    line2 = ax2.plot(df.index, df['PRECIP'], color=color, label='Precipitation')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.spines['right'].set_position(('outward', 60))  # Offset the right spine
    
    # Create a third y-axis for the third feature
    ax3 = ax1.twinx()
    color = 'tab:red'
    ax3.set_ylabel(third_feature, color=color)
    line3 = ax3.plot(df.index, df[third_feature], color=color, label=third_feature, linestyle='--')
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.spines['right'].set_position(('outward', 120))  # Offset the right spine further
    
    # Add a title
    time_range = f"from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"
    plt.title(f'Streamflow, Precipitation, and {third_feature} {time_range}')
    
    # Add legends - combine all lines and labels
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels, loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if output path is provided
    if not output_path:
        start_str = df.index[0].strftime('%Y%m%d')
        end_str = df.index[-1].strftime('%Y%m%d')
        output_path = f'imgs/streamflow_precip_{third_feature}_{start_str}_{end_str}.png'
    
    plt.savefig(output_path, dpi=300)
    print(f"Figure saved to {output_path}")
    
    # Also display the figure
    plt.show()

if __name__ == "__main__":
    #  python combined_visualize.py --start_date "1975-07-01" --end_date "1975-8-15" --third_feature "ET"
    #  python combined_visualize.py --start_date "1975-07-03" --end_date "1975-7-06" --third_feature "ET"
    # python combined_visualize.py --start_date "1985-07-01" --end_date "1985-8-15" --third_feature "ET"
    #  python combined_visualize.py --start_date "1985-07-03" --end_date "1985-7-06" --third_feature "ET"
    parser = argparse.ArgumentParser(description='Visualize streamflow and precipitation data')
    parser.add_argument('--csv_file', type=str, default='processed/KettleRiverModels_hist_scaled_combined.csv',
                        help='Path to CSV file with streamflow and precipitation data')
    parser.add_argument('--start_date', type=str, help='Start date in format YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, help='End date in format YYYY-MM-DD')
    parser.add_argument('--output', type=str, help='Output file path for the figure')
    parser.add_argument('--third_feature', type=str, help='Third feature to visualize (e.g., ET, PET, SNOW)')
    
    args = parser.parse_args()
    
    if args.third_feature:
        visualize_streamflow_precip_and_feature(
            args.csv_file,
            args.third_feature,
            args.start_date,
            args.end_date,
            args.output
        )
    else:
        visualize_streamflow_and_precip(
            args.csv_file,
            args.start_date,
            args.end_date,
            args.output
        )