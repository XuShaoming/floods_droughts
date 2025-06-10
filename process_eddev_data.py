"""
EDDEV (Environmental Data for Drought and Extreme Precipitation) Data Processing Script

Purpose:
    This script processes meteorological data from the EDDEV1 dataset for watershed analysis.
    It maps climate data grid points to watersheds and calculates area-weighted meteorological values.

Workflow:
    1. Loads watershed shapefiles and calculates centroids
    2. Reads meteorological data from CSV files (T2, DEWPT, PRECIP, SWDNB, WSPD10, LH)
    3. Finds the nearest grid points to each watershed centroid (using K-Nearest Neighbors)
    4. Calculates Inverse Distance Weighting (IDW) for climate data interpolation
    5. Processes data by time period and calculates area-weighted averages
    6. Outputs processed data to a CSV file with meteorological variables by watershed

Variables:
    - T2: Temperature at 2m (K)
    - DEWPT: Dew point temperature (K)
    - PRECIP: Precipitation (mm)
    - SWDNB: Downward shortwave radiation at ground surface (W/m²)
    - WSPD10: Wind speed at 10m (m/s)
    - LH: Latent heat flux (W/m²)

Notes:
    - This script uses KNN-IDW interpolation to map gridded climate data to watershed areas
    - Area weighting ensures proper representation of climate variables across watersheds
    - Pre-computed nearest grid points can be loaded from a .npy file to speed up processing
    
Usage:
    - To process a specific date range:
      python process_eddev_data.py --start "1975-01-01" --end "1975-01-31" --basin "KettleR_Watersheds" --scenario "Historical"
      python process_eddev_data.py --start "2025-05-05" --end "2025-05-7" --basin "KettleR_Watersheds" --scenario "RCP4.5"
      
    python process_eddev_data.py --all --basin "KettleR_Watersheds" --scenario "Historical"
    python process_eddev_data.py --all --basin "KettleR_Watersheds" --scenario "RCP4.5"
    python process_eddev_data.py --all --basin "KettleR_Watersheds" --scenario "RCP8.5"

    - To process all available data:
      python process_eddev_data.py --all
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import os
import tqdm
import pytz
import argparse
from datetime import datetime

# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process EDDEV meteorological data for watersheds')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true', help='Process all available data')
    group.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD), required when using --start')
    parser.add_argument('--basin', type=str, default='KettleR_Watersheds', help='Basin name (default: KettleR_Watersheds)')
    parser.add_argument('--scenario', type=str, default='Historical', help='Climate scenario (default: Historical). Options: Historical, RCP4.5, RCP8.5')
    args = parser.parse_args()
    
    # Validate that if start is provided, end is also provided
    if args.start and not args.end:
        parser.error("--end is required when --start is specified")
    
    return args

# File paths
base_dir = "eddev1"
centroid_locations_csv = f"{base_dir}/Lat_Lon_Centroid_Locations.csv"
meteo_list = ['T2', 'DEWPT', 'PRECIP', 'SWDNB', 'WSPD10', 'LH']
output_dir = "processed"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def main():
    args = parse_arguments()
    basin = args.basin
    scenario = args.scenario
    watersheds_shp = f"{base_dir}/{basin}_NewMetSeg.shp"

    # Determine time range based on command line arguments
    if args.all:
        # Will determine time range from data after loading
        time_start = None
        time_end = None
        output_file = f"{output_dir}/{basin}_{scenario}_eddev1.csv"
        print("Processing ALL available data...")
    else:
        # Use user-specified time range
        time_start = f"{args.start} 00:00:00"
        time_end = f"{args.end} 23:00:00"
        date_range_str = f"{args.start}_to_{args.end}"
        output_file = f"{output_dir}/{basin}_{scenario}_eddev1_{date_range_str}.csv"
        print(f"Processing time range: {time_start} to {time_end}")

    try:
        print("Loading watershed shapefile...")
        # Load Watershed Shapefile
        watersheds_gdf = gpd.read_file(watersheds_shp)
        print(f"Loaded {len(watersheds_gdf)} watersheds")
        
        # Reproject to a projected CRS (e.g., EPSG:5070 - USA Contiguous Albers Equal Area)
        watersheds_gdf = watersheds_gdf.to_crs(epsg=5070)
        # Calculate Centroids for Watersheds
        watersheds_gdf['centroid'] = watersheds_gdf.geometry.centroid

        # Load Centroid Locations CSV
        print("Loading centroid locations...")
        centroid_df = pd.read_csv(centroid_locations_csv)
        print(f"Loaded {len(centroid_df)} centroid locations")
        
        # Create GeoDataFrame for centroids and reproject to the same projected CRS
        centroid_gdf = gpd.GeoDataFrame(
            centroid_df,
            geometry=gpd.points_from_xy(centroid_df['lon'], centroid_df['lat']),
            crs="EPSG:4326"
        ).to_crs(epsg=5070)

        # Find the Nearest Grid Point for Each Watershed Centroid
        print("Finding nearest grid points...")
        if os.path.exists('nearest_grid_point_KettleR_Watersheds.npy'):
            nearest_grid_point = np.load('nearest_grid_point_KettleR_Watersheds.npy')
            print(f"Loaded existing nearest grid points from file")
        else:
            print("Computing nearest grid points (this may take a while)...")
            nearest_grid_point = []
            for centroid in watersheds_gdf['centroid']:
                distances = centroid_gdf.geometry.distance(centroid)
                nearest_idx = distances.idxmin()
                nearest_grid_point.append(centroid_df.loc[nearest_idx, 'Centroid_ID'])
            nearest_grid_point = np.array(nearest_grid_point)
            np.save('nearest_grid_point_KettleR_Watersheds.npy', nearest_grid_point)

        nearest_grid_point_set = set(nearest_grid_point)
        watersheds_gdf['Nearest_Grid_ID'] = nearest_grid_point
        
        print(f"Number of unique nearest grid points: {len(nearest_grid_point_set)}")

        # Calculate HUC8 area totals
        print("Calculating Watersheds area totals...")
        # # HUC8 area calculations if needed
        # huc8_area_totals = watersheds_gdf.groupby('HUC8')['Area_ac'].sum().reset_index()
        # watersheds_area_totals = watersheds_area_totals.rename(columns={'Area_ac': 'Area_ac_total'})
        # print(f"Watershed area totals:")
        # print(watersheds_area_totals)
        # # Merge HUC8 area totals back to watersheds_gdf
        # watersheds_gdf = pd.merge(watersheds_gdf, watersheds_area_totals, on='HUC8', how='left')

        watersheds_area_totals = watersheds_gdf['Area_ac'].sum()
        print(f"Watershed area totals:")
        print(watersheds_area_totals)
        # Add watersheds_area_totals to watersheds_gdf
        watersheds_gdf['Area_ac_total'] = watersheds_area_totals

        # Load meteorology data
        meteo_data = {}
        for meteo in meteo_list:
            meteo_csv = f"{base_dir}/WRF-CESM/{scenario}_{meteo}.csv"
            print(f"Loading {meteo} data from {meteo_csv}...")
            meteo_df = pd.read_csv(meteo_csv)
            meteo_df['Date'] = pd.to_datetime(meteo_df['Date'])
            meteo_data[meteo] = meteo_df
        
        # Determine the full date range from the data if processing all dates
        if args.all:
            # Find the min and max dates across all meteorological variables
            min_dates = []
            max_dates = []
            for meteo in meteo_list:
                min_dates.append(meteo_data[meteo]['Date'].min())
                max_dates.append(meteo_data[meteo]['Date'].max())
            
            time_start = min(min_dates)
            time_end = max(max_dates)
            print(f"Determined data range: {time_start} to {time_end}")
        else:
            # Convert string dates to datetime objects
            time_start = pd.to_datetime(time_start)
            time_end = pd.to_datetime(time_end)
        
        # Create time period range
        time_period = pd.date_range(start=time_start, end=time_end, freq='h')
        print(f"Processing {len(time_period)} time steps")

        # Calculate normalized weights for KNN-IDW interpolation
        print("Calculating KNN-IDW weights for each watershed...")
        watershed_weights = {}
        
        for idx, row in tqdm.tqdm(watersheds_gdf.iterrows(), total=len(watersheds_gdf)):
            huc12_id = row['SubID']
            centroid = row['centroid']
            weights = {}
            
            for grid_id in nearest_grid_point_set:
                grid_id_str = str(grid_id)
                grid_point_geom = centroid_gdf.loc[centroid_gdf['Centroid_ID'] == grid_id, 'geometry'].values[0]
                distance = centroid.distance(grid_point_geom)
                # Avoid zero distance by adding a small constant
                adjusted_distance = max(distance, 1e-6)
                # Calculate weight as inverse of distance squared
                weight = 1 / (adjusted_distance ** 2)
                weights[grid_id_str] = weight
            
            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            for grid_id in weights:
                weights[grid_id] = weights[grid_id] / total_weight
                
            watershed_weights[huc12_id] = weights

        # Create a list to store all results
        all_results = []

        # Process each time period with progress bar showing percentage
        print(f"Processing time periods from {time_start} to {time_end}...")
        for i, period in enumerate(tqdm.tqdm(time_period, desc="Processing time periods")):
            # period_ct = period.tz_localize("UTC").astimezone(timezone)
            period_ct = period
            
            # Process each meteorological variable
            huc8_meteo_values = {}
            
            for meteo in meteo_list:
                # Filter meteorology data for the current period
                period_meteo_df = meteo_data[meteo][meteo_data[meteo]['Date'] == period]
                
                if len(period_meteo_df) == 0:
                    continue
                    
                # Calculate weighted value for each HUC12 watershed using KNN-IDW
                huc12_meteo_values = []
                
                for _, huc12_row in watersheds_gdf.iterrows():
                    huc12_id = huc12_row['SubID']
                    weights = watershed_weights[huc12_id]
                    
                    # Calculate weighted value for this HUC12
                    weighted_value = 0
                    for grid_id, weight in weights.items():
                        if grid_id in period_meteo_df.columns:
                            grid_value = period_meteo_df[grid_id].values[0]
                            weighted_value += grid_value * weight
                        
                    huc12_meteo_values.append(weighted_value)
                
                # Store the HUC12 values for area-weighted averaging
                watersheds_gdf[f'{meteo}_value'] = huc12_meteo_values
                
                # Calculate area-weighted average for this HUC8
                huc8_meteo_values[meteo] = np.sum(watersheds_gdf[f'{meteo}_value'] * watersheds_gdf['Area_ac']) / watersheds_gdf['Area_ac_total'].iloc[0]
            
            # Skip this HUC8 at this time if we don't have all meteo variables
            if len(huc8_meteo_values) < len(meteo_list):
                continue
                
            # Create a result row for this HUC8 at this time
            result_row = {
                'Datetime': period_ct,
                'Area_ac_total': watersheds_gdf['Area_ac_total'].iloc[0],
            }
            
            # Add meteorological variables
            for meteo in meteo_list:
                result_row[meteo] = huc8_meteo_values[meteo]
                
            # Add results for this time period
            all_results.append(result_row)
            
            # # Periodically save intermediate results to avoid losing progress on long runs
            # if args.all and i > 0 and i % 10000 == 0:
            #     print(f"\nSaving intermediate results after processing {i} of {len(time_period)} time steps...")
            #     temp_results_df = pd.DataFrame(all_results)
            #     temp_results_df = temp_results_df.sort_values(['Datetime'])
            #     column_order = ['Datetime'] + meteo_list + ['Area_ac_total']
            #     temp_results_df = temp_results_df[column_order]
            #     temp_output_file = f"{output_file}.partial_{i}"
            #     temp_results_df.to_csv(temp_output_file, index=False)
            #     print(f"Intermediate results saved to {temp_output_file}")
        
        # Create DataFrame from results
        results_df = pd.DataFrame(all_results)
        
        if len(results_df) == 0:
            print("No results were generated. Check time range and data availability.")
            return
            
        # Sort by datetime and HUC8
        results_df = results_df.sort_values(['Datetime'])
        
        # Reorder columns to match desired format
        column_order = ['Datetime'] + meteo_list + ['Area_ac_total']
        results_df = results_df[column_order]
        
        # Save to CSV
        print(f"Saving results to {output_file}...")
        results_df.to_csv(output_file, index=False)
        print(f"Processing complete. Results saved to {output_file}")
        
        # Print sample of results
        print("\nSample of results:")
        print(results_df.head().to_string())
        
        # Print summary statistics
        print("\nSummary statistics:")
        print(results_df.describe())
        
    except Exception as e:
        import traceback
        print(f"Error processing data: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()