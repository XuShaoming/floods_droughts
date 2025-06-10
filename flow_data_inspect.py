import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from pathlib import Path

# Set plotting style
plt.style.use('ggplot')
sns.set_context("talk")

def load_flow_data(file_path):
    """
    Load flow data from CSV file with hourly data.
    
    Args:
        file_path: Path to the flow data CSV file
        
    Returns:
        DataFrame with parsed datetime and flow data
    """
    print(f"Loading flow data from {os.path.basename(file_path)}...")
    df = pd.read_csv(file_path, parse_dates=['Datetime'])
    df.set_index('Datetime', inplace=True)
    
    # Get the flow column name (may vary)
    flow_col = df.columns[0]
    
    # Basic statistics
    print(f"\nFlow Statistics (in {flow_col.split('_')[-1] if '_' in flow_col else 'units'}):")
    print(f"Min: {df[flow_col].min():.2f}")
    print(f"Max: {df[flow_col].max():.2f}")
    print(f"Mean: {df[flow_col].mean():.2f}")
    print(f"Median: {df[flow_col].median():.2f}")
    print(f"Standard Deviation: {df[flow_col].std():.2f}")
    
    return df

def load_daily_outputs(file_path):
    """
    Load daily output metrics from CSV file.
    
    Args:
        file_path: Path to the daily outputs CSV file
        
    Returns:
        DataFrame with parsed datetime and metrics
    """
    print(f"\nLoading daily outputs from {os.path.basename(file_path)}...")
    df = pd.read_csv(file_path, parse_dates=['Datetime'])
    df.set_index('Datetime', inplace=True)
    
    # Print column descriptions based on ReadMeFirst.txt
    print("\nDaily Output Metrics:")
    print("PET - Weighted Potential Evapotranspiration in inches")
    print("ET - Weighted actual Evapotranspiration in inches")
    print("SUPY - Weighted moisture supply (Rain + Snow Melt) in inches")
    print("SNOW - Weighted snow depth in feet")
    print("TWS - Weighted total soil moisture in inches")
    print("LZS - Weighted lower zone soil moisture in inches")
    print("AGW - Weighted active groundwater storage in inches")
    
    # Basic statistics for each metric
    print("\nBasic Statistics:")
    for col in df.columns:
        print(f"\n{col} Statistics:")
        print(f"Min: {df[col].min():.2f}")
        print(f"Max: {df[col].max():.2f}")
        print(f"Mean: {df[col].mean():.2f}")
        
    return df

def analyze_time_series(flow_df, output_dir):
    """
    Perform time series analysis of flow data.
    
    Args:
        flow_df: DataFrame containing flow data
        output_dir: Directory to save output plots
        
    Returns:
        Dictionary containing resampled flow data at different frequencies
    """
    print("\nPerforming time series analysis of flow data...")
    
    # Resample flow data to daily, weekly, monthly, and yearly
    flow_daily = flow_df.resample('D').mean()
    flow_weekly = flow_df.resample('W').mean()
    # Using 'ME' instead of 'M' as per deprecation warning
    flow_monthly = flow_df.resample('ME').mean()
    # Using 'YE' instead of 'Y' as per deprecation warning
    flow_yearly = flow_df.resample('YE').mean()
    
    # Plot flow time series
    plt.figure(figsize=(16, 8))
    plt.plot(flow_daily.index, flow_daily, color='tab:blue', label='Daily Mean Flow')
    plt.title('Kettle River Models Daily Mean Flow (Historical Scaled)')
    plt.xlabel('Date')
    plt.ylabel('Flow')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'kettle_river_daily_flow.png')
    
    # Add month and year columns for later analyses
    flow_daily['Month'] = flow_daily.index.month
    flow_daily['Year'] = flow_daily.index.year
    
    return {
        'daily': flow_daily,
        'weekly': flow_weekly,
        'monthly': flow_monthly,
        'yearly': flow_yearly
    }

def analyze_monthly_seasonality(flow_daily, output_dir):
    """
    Analyze monthly seasonal patterns in flow data.
    
    Args:
        flow_daily: DataFrame containing daily flow data with Month column
        output_dir: Directory to save output plots
    """
    print("\nAnalyzing seasonal patterns...")
    
    monthly_means = flow_daily.groupby('Month').mean()
    
    plt.figure(figsize=(14, 7))
    plt.bar(monthly_means.index, monthly_means.iloc[:, 0], color='tab:blue')
    plt.title('Kettle River Models Average Monthly Flow (Historical Scaled)')
    plt.xlabel('Month')
    plt.ylabel('Average Flow')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.tight_layout()
    plt.savefig(output_dir / 'kettle_river_monthly_avg_flow.png')

def analyze_correlations(daily_df, output_dir):
    """
    Analyze correlations between different daily metrics.
    
    Args:
        daily_df: DataFrame containing daily output metrics
        output_dir: Directory to save output plots
    """
    print("\nCalculating correlation between daily metrics...")
    correlation = daily_df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Daily Metrics (Historical Scaled)')
    plt.tight_layout()
    plt.savefig(output_dir / 'kettle_river_correlation_matrix.png')

def analyze_snow_flow_relationship(flow_daily, daily_df, output_dir):
    """
    Analyze the relationship between snow and flow.
    
    Args:
        flow_daily: DataFrame containing daily flow data
        daily_df: DataFrame containing daily output metrics including SNOW
        output_dir: Directory to save output plots
        
    Returns:
        DataFrame with combined flow and snow data
    """
    print("\nAnalyzing snow and flow relationship...")
    
    # Combine daily flow with snow data
    combined_df = pd.DataFrame()
    combined_df['Flow'] = flow_daily.iloc[:, 0]
    combined_df['SNOW'] = daily_df['SNOW']
    
    plt.figure(figsize=(14, 7))
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Flow', color='tab:blue')
    ax1.plot(combined_df.index, combined_df['Flow'], color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Snow Depth (feet)', color='tab:red')
    ax2.plot(combined_df.index, combined_df['SNOW'], color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    plt.title('Kettle River Daily Flow and Snow Depth Comparison (Historical Scaled)')
    fig.tight_layout()
    plt.savefig(output_dir / 'kettle_river_flow_snow_comparison.png')
    
    return combined_df

def analyze_annual_variability(flow_daily, output_dir):
    """
    Analyze annual variability in flow data.
    
    Args:
        flow_daily: DataFrame containing daily flow data with Year column
        output_dir: Directory to save output plots
    """
    print("\nAnalyzing annual variability...")
    # Get the name of the flow column (first column in the DataFrame)
    flow_col = flow_daily.columns[0]
    
    # Calculate yearly statistics
    yearly_stats = flow_daily.groupby('Year')[flow_col].agg(['mean', 'min', 'max', 'std'])
    
    # Plot annual statistics
    plt.figure(figsize=(16, 10))
    plt.plot(yearly_stats.index, yearly_stats['mean'], marker='o', label='Mean Flow')
    plt.fill_between(yearly_stats.index, 
                     yearly_stats['mean'] - yearly_stats['std'],
                     yearly_stats['mean'] + yearly_stats['std'],
                     alpha=0.3, label='± 1 Std Dev')
    plt.plot(yearly_stats.index, yearly_stats['max'], marker='^', linestyle='--', label='Max Flow')
    plt.plot(yearly_stats.index, yearly_stats['min'], marker='v', linestyle='--', label='Min Flow')
    
    plt.title('Kettle River Annual Flow Statistics (Historical Scaled)')
    plt.xlabel('Year')
    plt.ylabel('Flow')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / 'kettle_river_annual_variability.png')

# Check differences between flow_daily and daily_df
def check_differences(flow_daily, daily_df):
    """
    Check differences between flow_daily and daily_df.
    
    Args:
        flow_daily: DataFrame containing daily flow data
        daily_df: DataFrame containing daily output metrics
        
    Returns:
        DataFrame highlighting differences in indices and missing data
    """
    print("\nChecking differences between flow_daily and daily_df...")
    
    # Find indices present in one DataFrame but not the other
    flow_only = flow_daily.index.difference(daily_df.index)
    daily_only = daily_df.index.difference(flow_daily.index)
    
    # Report missing indices
    print(f"Indices in flow_daily but not in daily_df: {len(flow_only)}")
    print(f"Indices in daily_df but not in flow_daily: {len(daily_only)}")
    
    # Create a summary DataFrame
    differences = pd.DataFrame({
        'In_flow_daily_not_in_daily_df': flow_only,
        'In_daily_df_not_in_flow_daily': daily_only
    })
    
    return differences

if __name__ == "__main__":
    """
    Analyze Kettle River Models flow data and daily outputs.
    """
    # Define paths to data files
    base_path = Path("/home/kumarv/xu000114/floods_droughts/flow_data")
    flow_file = base_path / "KettleRiverModels_outlet_hist_scaled_FLOW.csv"
    daily_file = base_path / "KettleRiverModels_outlet_hist_scaled_Daily_outputs.csv"
    
    # Create output directory for plots
    output_dir = Path("/home/kumarv/xu000114/floods_droughts/flow_data_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    flow_df = load_flow_data(flow_file)
    daily_df = load_daily_outputs(daily_file)
    
    # Perform different analyses
    flow_resampled = analyze_time_series(flow_df, output_dir)
    flow_daily = flow_resampled['daily']
    
    analyze_monthly_seasonality(flow_daily, output_dir)
    analyze_correlations(daily_df, output_dir)
    combined_df = analyze_snow_flow_relationship(flow_daily, daily_df, output_dir)
    analyze_annual_variability(flow_daily, output_dir)
    
    differences = check_differences(flow_daily, daily_df)
    print("\nDifferences summary:")
    print(differences)
    
    print("\nAnalysis complete. Results saved to:", output_dir)