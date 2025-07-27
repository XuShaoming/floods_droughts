"""
 EDDEV Data Processing Script
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import os
import tqdm
import pytz
import argparse
from datetime import datetime

# File paths (same as original)
base_dir = os.path.join("data","eddev1")
centroid_locations_csv = f"{base_dir}/Lat_Lon_Centroid_Locations.csv"
meteo_list = ['T2', 'DEWPT', 'PRECIP', 'SWDNB', 'WSPD10', 'LH']
output_dir = "data_processed"
os.makedirs(output_dir, exist_ok=True)

basin_names ={
    'WatonwanR_Watersheds': 'Watonwan',
    'LeSueurR_Watersheds': 'LeSueur',
    'KettleR_Watersheds': 'KettleRiverModels',
    'BlueEarthR_Watersheds': 'BlueEarth'
}

scenario_names = {
    'Historical': 'hist_scaled',
    'RCP4.5': 'RCP4.5',
    'RCP8.5': 'RCP8.5'
}

# Parse command line arguments (same as original)
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process EDDEV meteorological data for watersheds')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true', help='Process all available data')
    group.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD), required when using --start')
    parser.add_argument('--basin', type=str, default='KettleR_Watersheds', help='Basin name (default: KettleR_Watersheds)')
    parser.add_argument('--scenario', type=str, default='Historical', help='Climate scenario (default: Historical). Options: Historical, RCP4.5, RCP8.5')
    args = parser.parse_args()
    
    if args.start and not args.end:
        parser.error("--end is required when --start is specified")
    
    return args

def main():
    args = parse_arguments()
    basin = args.basin
    scenario = args.scenario
    watersheds_shp = f"{base_dir}/{basin}_NewMetSeg.shp"

    # Determine time range and output file (same as original)
    if args.all:
        time_start = None
        time_end = None
        output_file = f"{output_dir}/{basin_names[basin]}/{basin_names[basin]}_{scenario_names[scenario]}_eddev1.csv"
        print("Processing ALL available data...")
    else:
        time_start = f"{args.start} 00:00:00"
        time_end = f"{args.end} 23:00:00"
        date_range_str = f"{args.start}_to_{args.end}"
        output_file = f"{output_dir}/{basin_names[basin]}/{basin_names[basin]}_{scenario_names[scenario]}_eddev1_{date_range_str}.csv"
        print(f"Processing time range: {time_start} to {time_end}")

    try:
        # Load spatial data (same as original)
        print("Loading watershed shapefile...")
        watersheds_gdf = gpd.read_file(watersheds_shp)
        print(f"Loaded {len(watersheds_gdf)} watersheds")
        
        watersheds_gdf = watersheds_gdf.to_crs(epsg=5070)
        watersheds_gdf['centroid'] = watersheds_gdf.geometry.centroid

        print("Loading centroid locations...")
        centroid_df = pd.read_csv(centroid_locations_csv)
        print(f"Loaded {len(centroid_df)} centroid locations")
        
        centroid_gdf = gpd.GeoDataFrame(
            centroid_df,
            geometry=gpd.points_from_xy(centroid_df['lon'], centroid_df['lat']),
            crs="EPSG:4326"
        ).to_crs(epsg=5070)

        # Find nearest grid points (same as original)
        print("Finding nearest grid points...")
        nearest_grid_point_path = os.path.join(output_dir, basin_names[basin], f'nearest_grid_point_{basin_names[basin]}.npy')
        if os.path.exists(nearest_grid_point_path):
            nearest_grid_point = np.load(nearest_grid_point_path)
            print(f"Loaded existing nearest grid points from file {nearest_grid_point_path}")
        else:
            print("Computing nearest grid points (this may take a while)...")
            nearest_grid_point = []
            for centroid in watersheds_gdf['centroid']:
                distances = centroid_gdf.geometry.distance(centroid)
                nearest_idx = distances.idxmin()
                nearest_grid_point.append(centroid_df.loc[nearest_idx, 'Centroid_ID'])
            nearest_grid_point = np.array(nearest_grid_point)
            np.save(nearest_grid_point_path, nearest_grid_point)

        nearest_grid_point_set = set(nearest_grid_point)
        watersheds_gdf['Nearest_Grid_ID'] = nearest_grid_point
        print(f"Number of unique nearest grid points: {len(nearest_grid_point_set)}")

        # Calculate area totals (same as original)
        print("Calculating Watersheds area totals...")
        watersheds_area_totals = watersheds_gdf['Area_ac'].sum()
        watersheds_gdf['Area_ac_total'] = watersheds_area_totals

        # OPTIMIZATION 1: Load and index meteorological data efficiently
        print("Loading and indexing meteorological data...")
        meteo_data = {}
        for meteo in meteo_list:
            meteo_csv = f"{base_dir}/WRF-CESM/{scenario}_{meteo}.csv"
            print(f"Loading {meteo} data from {meteo_csv}...")
            meteo_df = pd.read_csv(meteo_csv)
            meteo_df['Date'] = pd.to_datetime(meteo_df['Date'])
            
            # OPTIMIZATION: Set datetime as index for fast lookups
            meteo_df.set_index('Date', inplace=True)
            meteo_data[meteo] = meteo_df
        
        # Determine time range from data if processing all dates
        if args.all:
            min_dates = [meteo_data[meteo].index.min() for meteo in meteo_list]
            max_dates = [meteo_data[meteo].index.max() for meteo in meteo_list]
            time_start = min(min_dates)
            time_end = max(max_dates)
            print(f"Determined data range: {time_start} to {time_end}")
        else:
            time_start = pd.to_datetime(time_start)
            time_end = pd.to_datetime(time_end)
        
        time_period = pd.date_range(start=time_start, end=time_end, freq='h')
        print(f"Processing {len(time_period)} time steps")

        # OPTIMIZATION 2: Pre-compute weights once
        print("Pre-computing KNN-IDW weights...")
        watershed_weights = {}
        
        # Convert to numpy arrays for faster processing
        watershed_ids = watersheds_gdf['SubID'].values
        watershed_areas = watersheds_gdf['Area_ac'].values
        
        for idx, row in tqdm.tqdm(watersheds_gdf.iterrows(), total=len(watersheds_gdf), desc="Computing weights"):
            huc12_id = row['SubID']
            centroid = row['centroid']
            weights = {}
            
            for grid_id in nearest_grid_point_set:
                grid_id_str = str(grid_id)
                grid_point_geom = centroid_gdf.loc[centroid_gdf['Centroid_ID'] == grid_id, 'geometry'].values[0]
                distance = centroid.distance(grid_point_geom)
                adjusted_distance = max(distance, 1e-6)
                weight = 1 / (adjusted_distance ** 2)
                weights[grid_id_str] = weight
            
            # Normalize weights
            total_weight = sum(weights.values())
            weights = {k: v/total_weight for k, v in weights.items()}
            watershed_weights[huc12_id] = weights

        # Process time periods
        all_results = []
        print(f"Processing time periods from {time_start} to {time_end}...")
        
        for i, period in enumerate(tqdm.tqdm(time_period, desc="Processing time periods")):
            huc8_meteo_values = {}
            
            for meteo in meteo_list:
                meteo_df = meteo_data[meteo]
                
                # OPTIMIZATION 3: Fast index-based lookup instead of filtering
                if period not in meteo_df.index:
                    continue
                    
                period_meteo_data = meteo_df.loc[period]
                
                # Calculate weighted values for all watersheds
                huc12_meteo_values = np.zeros(len(watershed_ids))
                
                for j, huc12_id in enumerate(watershed_ids):
                    weights = watershed_weights[huc12_id]
                    
                    weighted_value = 0
                    for grid_id, weight in weights.items():
                        if grid_id in period_meteo_data:
                            weighted_value += period_meteo_data[grid_id] * weight
                        
                    huc12_meteo_values[j] = weighted_value
                
                # OPTIMIZATION 4: Vectorized area-weighted average
                huc8_meteo_values[meteo] = np.sum(huc12_meteo_values * watershed_areas) / watersheds_area_totals
            
            # Skip if incomplete data
            if len(huc8_meteo_values) < len(meteo_list):
                continue
                
            # Create result row
            result_row = {
                'Datetime': period,
                'Area_ac_total': watersheds_area_totals,
            }
            result_row.update(huc8_meteo_values)
            all_results.append(result_row)
        
        # Create and save results (same as original)
        results_df = pd.DataFrame(all_results)
        
        if len(results_df) == 0:
            print("No results were generated. Check time range and data availability.")
            return
            
        results_df = results_df.sort_values(['Datetime'])
        column_order = ['Datetime'] + meteo_list + ['Area_ac_total']
        results_df = results_df[column_order]
        
        print(f"Saving results to {output_file}...")
        results_df.to_csv(output_file, index=False)
        print(f"Processing complete. Results saved to {output_file}")
        
        print("\nSample of results:")
        print(results_df.head().to_string())
        
        print("\nSummary statistics:")
        print(results_df.describe())
        
    except Exception as e:
        import traceback
        print(f"Error processing data: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
