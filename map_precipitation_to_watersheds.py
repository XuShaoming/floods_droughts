import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import matplotlib.pyplot as plt

# File paths
base_dir = "eddev1"
watersheds_shp = f"{base_dir}/KettleR_Watersheds_NewMetSeg.shp"
rcp4_5_precip_csv = f"{base_dir}/WRF-CESM/RCP4.5_PRECIP.csv"
centroid_locations_csv = f"{base_dir}/Lat_Lon_Centroid_Locations.csv"

# Load Watershed Shapefile
watersheds_gdf = gpd.read_file(watersheds_shp)
# Reproject to a projected CRS (e.g., EPSG:5070 - USA Contiguous Albers Equal Area)
watersheds_gdf = watersheds_gdf.to_crs(epsg=5070)

# Load RCP4.5 Precipitation CSV
precip_df = pd.read_csv(rcp4_5_precip_csv)
precip_df['Date'] = pd.to_datetime(precip_df['Date'])  # Ensure datetime format

# Load Centroid Locations CSV
centroid_df = pd.read_csv(centroid_locations_csv)
# Create GeoDataFrame for centroids and reproject to the same projected CRS
centroid_gdf = gpd.GeoDataFrame(
    centroid_df,
    geometry=gpd.points_from_xy(centroid_df['lon'], centroid_df['lat']),
    crs="EPSG:4326"
).to_crs(epsg=5070)

# Calculate Centroids for Watersheds
watersheds_gdf['centroid'] = watersheds_gdf.geometry.centroid

# Find the Nearest Grid Point for Each Watershed Centroid
nearest_grid_point = []
for centroid in watersheds_gdf['centroid']:
    distances = centroid_gdf.geometry.distance(centroid)
    nearest_idx = distances.idxmin()
    nearest_grid_point.append(centroid_df.loc[nearest_idx, 'Centroid_ID'])

watersheds_gdf['Nearest_Grid_ID'] = nearest_grid_point

# Map Precipitation Data to Watersheds
# Average precipitation for nearest grid points for each watershed
precip_columns = precip_df.columns[1:]  # Exclude 'Date'
watersheds_precip = []
for _, row in watersheds_gdf.iterrows():
    grid_id = str(row['Nearest_Grid_ID'])
    if grid_id in precip_columns:
        watershed_precip = precip_df[grid_id].mean()  # Mean precipitation over time
        watersheds_precip.append(watershed_precip)
    else:
        watersheds_precip.append(None)

watersheds_gdf['Average_Precipitation'] = watersheds_precip

# Visualization
plt.figure(figsize=(12, 10))
ax = plt.gca()
watersheds_gdf.to_crs(epsg=4326).plot(
    ax=ax,
    column='Average_Precipitation',
    cmap='coolwarm',
    legend=True,
    edgecolor='black'
)
plt.title("Kettle River Watersheds with Average Precipitation (RCP4.5)", fontsize=16)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.savefig('KettleR_Watersheds_Avg_Precipitation.png')
plt.show()
