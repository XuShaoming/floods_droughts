import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os
import tqdm 
import numpy as np
from PIL import Image
import io
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


time_start = '1975-03-01 00:00:00'
time_end = '1975-03-10 00:00:00'
time_period = pd.date_range(start=time_start, end=time_end, freq='h')
# Define the Central Time timezone
timezone = pytz.timezone("US/Central")

for cmap, meteo in tqdm.tqdm(zip(cmaps, meteo_list)):
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
    nearest_grid_point = []
    for centroid in watersheds_gdf['centroid']:
        distances = centroid_gdf.geometry.distance(centroid)
        nearest_idx = distances.idxmin()
        nearest_grid_point.append(centroid_df.loc[nearest_idx, 'Centroid_ID'])

    watersheds_gdf['Nearest_Grid_ID'] = nearest_grid_point

    # Generate maps for each time period
    gif_images = []
    # Load meteorology CSV
    meteo_df = pd.read_csv(meteo_csv)
    meteo_df['Date'] = pd.to_datetime(meteo_df['Date'])  # Ensure datetime format

    # Filter meteorology data for the selected period
    selected_meteo_df = meteo_df[meteo_df['Date'].isin(time_period)]

    # Calculate vmin and vmax
    if meteo == 'PRECIP':
        vmin = 0
        vmax = np.percentile(selected_meteo_df.iloc[:, 1:].values.flatten(), 95)
    else:
        vmin = selected_meteo_df.iloc[:, 1:].min().min()
        vmax = selected_meteo_df.iloc[:, 1:].max().max()
    print(f"vmin = {vmin}, vmax = {vmax}")

    gif_images_pillow = []
    for period in tqdm.tqdm(time_period):
        period_ct = period.tz_localize("UTC").astimezone(timezone)

        # Filter meteorology data for the current period
        period_meteo_df = meteo_df[meteo_df['Date'] == period]

        # Map meteorology data to watersheds
        period_meteo = []
        for _, row in watersheds_gdf.iterrows():
            grid_id = str(row['Nearest_Grid_ID'])
            if grid_id in period_meteo_df.columns:
                period_meteo.append(period_meteo_df[grid_id].values[0])
            else:
                period_meteo.append(None)
        
        watersheds_gdf['meteorology'] = period_meteo

        # Plot map for the current period
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
        plt.title(f"Kettle River Watersheds -  {meteo} ({period_ct.strftime('%Y-%m-%d %H:%M:%S %Z')})", fontsize=16)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True)

        # Save the plot to a BytesIO buffer instead of disk
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        # Add the in-memory image to the list for the GIF
        gif_images_pillow.append(Image.open(buf))

    # Save the GIF with each frame running for 10 seconds
    gif_path = os.path.join(output_dir, f'KettleR_Watersheds_{meteo}.gif')
    gif_images_pillow[0].save(
        gif_path,
        save_all=True,
        append_images=gif_images_pillow[1:],
        duration=500,  # Frame duration in milliseconds (1 seconds)
        loop=0  # Infinite loop
    )

    print(f"GIF saved at {gif_path} with each frame displayed for 1 seconds")