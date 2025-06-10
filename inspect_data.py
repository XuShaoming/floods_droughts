import geopandas as gpd
import pandas as pd
import numpy as np
import os

# File paths
base_dir = "eddev1"
watersheds_shp = f"{base_dir}/KettleR_Watersheds_NewMetSeg.shp"
centroid_locations_csv = f"{base_dir}/Lat_Lon_Centroid_Locations.csv"
meteo_var = 'T2'  # Just check one variable to start

print("Data Inspection Script")
print("=====================")

# Check watersheds shapefile
print("\nChecking watersheds shapefile...")
watersheds_gdf = gpd.read_file(watersheds_shp)
print(f"Loaded {len(watersheds_gdf)} watersheds")
print(f"Columns: {', '.join(watersheds_gdf.columns)}")
print(f"HUC8 unique values: {', '.join(map(str, watersheds_gdf['HUC8'].unique()))}")
print(f"Sample data:")
print(watersheds_gdf[['HUC8', 'Area_ac']].head())

# Check centroid locations
print("\nChecking centroid locations...")
centroid_df = pd.read_csv(centroid_locations_csv)
print(f"Loaded {len(centroid_df)} centroid locations")
print(f"Columns: {', '.join(centroid_df.columns)}")
print(f"Sample data:")
print(centroid_df.head())

# Check meteorology data
print("\nChecking meteorology data...")
meteo_csv = f"{base_dir}/WRF-CESM/Historical_{meteo_var}.csv"
try:
    meteo_df = pd.read_csv(meteo_csv)
    print(f"Loaded data from {meteo_csv}")
    print(f"Shape: {meteo_df.shape}")
    print(f"Columns: {', '.join(meteo_df.columns[:5])}{'...' if len(meteo_df.columns) > 5 else ''}")
    
    # Convert Date column to datetime
    meteo_df['Date'] = pd.to_datetime(meteo_df['Date'])
    print(f"Time range: {meteo_df['Date'].min()} to {meteo_df['Date'].max()}")
    print(f"Sample data:")
    print(meteo_df.head(3).to_string(max_cols=8))
    
    # Check if grid IDs in meteo data match with centroid IDs
    meteo_cols = set(meteo_df.columns)
    centroid_ids = set(map(str, centroid_df['Centroid_ID']))
    print(f"\nCentroid IDs in meteo data: {len(meteo_cols.intersection(centroid_ids))}")
    print(f"Total centroid IDs: {len(centroid_ids)}")
    
except Exception as e:
    print(f"Error loading meteorology data: {str(e)}")

print("\nInspection complete.")
