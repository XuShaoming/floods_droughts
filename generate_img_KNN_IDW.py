import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os
import tqdm 
import numpy as np
from PIL import Image
import pytz

# File paths
base_dir = "eddev1"
watersheds_shp = f"{base_dir}/KettleR_Watersheds_NewMetSeg.shp"
centroid_locations_csv = f"{base_dir}/Lat_Lon_Centroid_Locations.csv"
meteo_list = ['T2', 'DEWPT', 'PRECIP', 'SWDNB', 'WSPD10']
cmaps = ['coolwarm', 'coolwarm', 'Blues', 'YlOrRd', 'viridis']
output_dir = "maps"
# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Selected time for visualization
selected_time = '1975-03-05 12:00:00'
selected_time = pd.Timestamp(selected_time)
# Define the Central Time timezone
timezone = pytz.timezone("US/Central")
selected_time_ct = selected_time.tz_localize("UTC").astimezone(timezone)

for cmap, meteo in zip(cmaps, meteo_list):
    meteo_csv = f"{base_dir}/WRF-CESM/Histmodel_{meteo}.csv"
    # Load Watershed Shapefile
    watersheds_gdf = gpd.read_file(watersheds_shp)
    # Reproject to a projected CRS (e.g., EPSG:5070 - USA Contiguous Albers Equal Area)
    watersheds_gdf = watersheds_gdf.to_crs(epsg=5070)
    # Calculate Centroids for Watersheds
    watersheds_gdf['centroid'] = watersheds_gdf.geometry.centroid

    # Load Centroid Locations CSV
    centroid_df = pd.read_csv(centroid_locations_csv)
    # Create GeoDataFrame for centroids and reproject to the same projected CRS
    centroid_gdf = gpd.GeoDataFrame(
        centroid_df,
        geometry=gpd.points_from_xy(centroid_df['lon'], centroid_df['lat']),
        crs="EPSG:4326"
    ).to_crs(epsg=5070)

    # Find the Nearest Grid Point for Each Watershed Centroid
    if os.path.exists('nearest_grid_point_KettleR_Watersheds.npy'):
        nearest_grid_point = np.load('nearest_grid_point_KettleR_Watersheds.npy')
    else:
        nearest_grid_point = []
        for centroid in watersheds_gdf['centroid']:
            distances = centroid_gdf.geometry.distance(centroid)
            nearest_idx = distances.idxmin()
            nearest_grid_point.append(centroid_df.loc[nearest_idx, 'Centroid_ID'])
        nearest_grid_point = np.array(nearest_grid_point)
        np.save('nearest_grid_point_KettleR_Watersheds.npy', nearest_grid_point)
    
    nearest_grid_point_set = set(nearest_grid_point)
    watersheds_gdf['Nearest_Grid_ID'] = nearest_grid_point
    # Load meteorology CSV
    meteo_df = pd.read_csv(meteo_csv)
    meteo_df['Date'] = pd.to_datetime(meteo_df['Date'])  # Ensure datetime format
    # Filter meteorology data for the current period
    period_meteo_df = meteo_df[meteo_df['Date'] == selected_time]
    # Map meteorology data to watersheds
    period_meteo = []
    for _, row in watersheds_gdf.iterrows():
        centroid = row['centroid']
        weights = []
        values = []
        for grid_id in nearest_grid_point_set:
            grid_point_geom = centroid_gdf.loc[centroid_gdf['Centroid_ID'] == grid_id, 'geometry'].values[0]
            distance = centroid.distance(grid_point_geom)
            # Avoid zero distance by adding a small constant (e.g., 1e-6)
            adjusted_distance = max(distance, 1e-6)
            # Append the weight as the inverse of distance
            weights.append(1 / adjusted_distance)
            # Append the corresponding value
            values.append(period_meteo_df[str(grid_id)].values[0])

        # Normalize weights to sum to 1
        weights = np.array(weights)
        normalized_weights = weights / weights.sum()
        # Compute the weighted average
        period_meteo.append(np.dot(normalized_weights, values))

    watersheds_gdf['meteorology'] = period_meteo

    # Calculate vmin and vmax
    if meteo == 'PRECIP':
        vmin = 0
        vmax = np.percentile(period_meteo_df.iloc[:, 1:].values.flatten(), 95)
    else:
        vmin = period_meteo_df.iloc[:, 1:].min().min()
        vmax = period_meteo_df.iloc[:, 1:].max().max()
    print(f"vmin = {vmin}, vmax = {vmax}")

    # Plot map for the selected time
    fig, ax = plt.subplots(figsize=(12, 10))
    ax = plt.gca()
    watersheds_gdf.to_crs(epsg=4326).plot(
        ax=ax,
        column='meteorology',
        cmap=cmap,
        legend=True,
        edgecolor='black',
        vmin=vmin,
        vmax=vmax
    )
    plt.title(f"Kettle River Watersheds - {meteo} ({selected_time_ct.strftime('%Y-%m-%d %H:%M:%S %Z')})", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)

    # Save the plot as a PNG file
    output_path = os.path.join(output_dir, f'KettleR_Watersheds_{meteo}_{selected_time_ct.strftime("%Y%m%d_%H%M")}.png')
    plt.tight_layout()
    plt.savefig(output_path, format='png')
    plt.close(fig)

    print(f"Image saved at {output_path}")
