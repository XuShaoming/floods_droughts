import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

'''
This script visualize:
    1. CLimate Data Ploygons with USA Contours
    2. Climate Data Locations with USA countours
    3. Kettle River Watershed in HUC8 and HUC12 levels. 
'''

# File paths
base_dir = "eddev1"
countries_shp = f"{base_dir}/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"
world = gpd.read_file(countries_shp)
# Filter for mainland USA (excluding Alaska and Hawaii)
usa = world[(world['ADMIN'] == 'United States of America') & 
            (world['SUBUNIT'] != 'Alaska') & 
            (world['SUBUNIT'] != 'Hawaii')]

# Visualization of KettleR_Watersheds_NewMetSeg.shp
watersheds_shp = f"{base_dir}/KettleR_Watersheds_NewMetSeg.shp"
watersheds_gdf = gpd.read_file(watersheds_shp)
# Reproject Watersheds to WGS84
watersheds_gdf_wgs84 = watersheds_gdf.to_crs(epsg=4326)
# Visualization of KettleR_Watersheds_NewMetSeg.shp
plt.figure(figsize=(10, 8))
ax = plt.gca()
# Plot Watersheds
watersheds_gdf_wgs84.plot(ax=ax, edgecolor='blue', facecolor='lightblue', alpha=0.5, label='Kettle River Watersheds')
# Custom Legend
custom_legend = [
    Patch(edgecolor='blue', facecolor='lightblue', alpha=0.5, label='Kettle River Watersheds'),
]
plt.legend(handles=custom_legend, loc='upper right')
plt.title("Kettle River Watersheds", fontsize=14)
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)
plt.grid(True)
plt.savefig('maps/KettleR_Watersheds_NewMetSeg.png')
plt.show()

# Visualization of Climate_Data_Locations.shp
locations_shp = f"{base_dir}/Climate_Data_Locations.shp"
locations_gdf = gpd.read_file(locations_shp)
# Reproject to WGS84
locations_gdf_wgs84 = locations_gdf.to_crs(epsg=4326)
# Plot Climate Data Locations with USA Contours
plt.figure(figsize=(10, 8))
ax = plt.gca()
usa.boundary.plot(ax=ax, color='black', linewidth=1, label='USA Boundary')
locations_gdf_wgs84.plot(ax=ax, color='red', markersize=1, alpha=0.7, label='Climate Data Locations')
watersheds_gdf_wgs84.plot(ax=ax, edgecolor='blue', facecolor='lightblue', alpha=0.5, label='Kettle River Watersheds')
# Custom legend
custom_legend = [
    Patch(edgecolor='green', facecolor='none', label='Climate Data Locations'),
    Patch(edgecolor='black', facecolor='none', label='USA Boundary'),
    Patch(edgecolor='blue', facecolor='lightblue', alpha=0.5, label='Kettle River Watersheds'),
]
plt.title("Climate Data Locations with USA Contours", fontsize=14)
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)
plt.legend(handles=custom_legend, loc='upper right')
plt.grid(True)
plt.savefig('maps/Climate_Data_Locations_with_USA_Contours.png')

# Visualization of Climate_Data_Polygons.shp
polygons_shp = f"{base_dir}/Climate_Data_Polygons.shp"
polygons_gdf = gpd.read_file(polygons_shp)
# Reproject polygons to WGS84
polygons_gdf_wgs84 = polygons_gdf.to_crs(epsg=4326)
# Visualization of Climate_Data_Polygons.shp with USA Contours
plt.figure(figsize=(10, 8))
ax = plt.gca()
usa.boundary.plot(ax=ax, color='black', linewidth=1, label='USA Boundary')
polygons_gdf_wgs84.plot(ax=ax, edgecolor='green', facecolor='none', label='Climate Data Polygons')
watersheds_gdf_wgs84.plot(ax=ax, edgecolor='blue', facecolor='lightblue', alpha=0.5, label='Kettle River Watersheds')
plt.title("Climate Data Polygons with USA Contours", fontsize=14)
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)
# Custom legend
custom_legend = [
    Patch(edgecolor='green', facecolor='none', label='Climate Data Polygons'),
    Patch(edgecolor='black', facecolor='none', label='USA Boundary'),
    Patch(edgecolor='blue', facecolor='lightblue', alpha=0.5, label='Kettle River Watersheds'),
]
# plt.legend()
plt.legend(handles=custom_legend, loc='upper right')
plt.grid(True)
plt.savefig('maps/Climate_Data_Polygons_with_USA_Contours.png')
plt.show()