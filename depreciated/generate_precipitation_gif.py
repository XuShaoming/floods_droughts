import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os
import tqdm 
import numpy as np
from PIL import Image

# File paths
base_dir = "eddev1"
watersheds_shp = f"{base_dir}/KettleR_Watersheds_NewMetSeg.shp"
centroid_locations_csv = f"{base_dir}/Lat_Lon_Centroid_Locations.csv"
precip_csv = f"{base_dir}/WRF-CESM/Histmodel_PRECIP.csv"
output_dir = "maps"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

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
time_start = '1975-02-01 00:00:00'
time_end = '1975-02-01 12:00:00'
time_period = pd.date_range(start=time_start, end=time_end, freq='h')

# Load Precipitation CSV
precip_df = pd.read_csv(precip_csv)
precip_df['Date'] = pd.to_datetime(precip_df['Date'])  # Ensure datetime format

# Filter precipitation data for the selected period
selected_precip_df = precip_df[precip_df['Date'].isin(time_period)]

# Calculate vmin and vmax
# vmin = selected_precip_df.iloc[:, 1:].min().min()
# vmax = selected_precip_df.iloc[:, 1:].max().max()
vmin = 0
vmax = np.percentile(selected_precip_df.iloc[:, 1:].values.flatten(), 95)
print(f"vmin = {vmin}, vmax = {vmax}")

for period in tqdm.tqdm(time_period):
    # Filter precipitation data for the current period
    period_precip_df = precip_df[precip_df['Date'] == period]

    # Map precipitation data to watersheds
    period_precip = []
    for _, row in watersheds_gdf.iterrows():
        grid_id = str(row['Nearest_Grid_ID'])
        if grid_id in period_precip_df.columns:
            period_precip.append(period_precip_df[grid_id].values[0])
        else:
            period_precip.append(None)
    
    watersheds_gdf['Precipitation'] = period_precip

    # Plot map for the current period
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    watersheds_gdf.to_crs(epsg=4326).plot(
        ax=ax,
        column='Precipitation',
        cmap='Blues',
        legend=True,
        edgecolor='black',
        vmin=vmin,
        vmax=vmax
    )
    plt.title(f"Kettle River Watersheds -  Precipitation ({period})", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)

    # Save the map as an image
    map_path = os.path.join(output_dir, f"precipitation_map_{period}.png")
    plt.savefig(map_path)
    gif_images.append(map_path)
    plt.close()


# Create GIF using Pillow
gif_images_pillow = [Image.open(img_path) for img_path in gif_images]

# Save the GIF with each frame running for 10 seconds
gif_path = 'KettleR_Watersheds_Precipitation.gif'
gif_images_pillow[0].save(
    gif_path,
    save_all=True,
    append_images=gif_images_pillow[1:],
    duration=1000,  # Frame duration in milliseconds (10 seconds)
    loop=0  # Infinite loop
)

print(f"GIF saved at {gif_path} with each frame displayed for 10 seconds")