import geopandas as gpd
import pandas as pd
import os

# Path to the unzipped directory
base_dir = "eddev1"

# List of shapefiles to inspect
shapefiles = [
    "Climate_Data_Locations.shp",
    "Climate_Data_Polygons.shp",
    "KettleR_Watersheds_NewMetSeg.shp",
]

# List of CSV files to inspect
csv_files = [
    "Lat_Lon_Centroid_Locations.csv",
    "WRF-CESM/Histmodel_DEWPT.csv",
    "WRF-CESM/Histmodel_PRECIP.csv",
    "WRF-CESM/Histmodel_SWDNB.csv",
    "WRF-CESM/Histmodel_T2.csv",
    "WRF-CESM/Histmodel_WSPD10.csv",
    "WRF-CESM/RCP4.5_DEWPT.csv",
    "WRF-CESM/RCP4.5_PRECIP.csv",
    "WRF-CESM/RCP4.5_SWDNB.csv",
    "WRF-CESM/RCP4.5_T2.csv",
    "WRF-CESM/RCP4.5_WSPD10.csv",
    "WRF-CESM/RCP6.0_DEWPT.csv",
    "WRF-CESM/RCP6.0_PRECIP.csv",
    "WRF-CESM/RCP6.0_SWDNB.csv",
    "WRF-CESM/RCP6.0_T2.csv",
    "WRF-CESM/RCP6.0_WSPD10.csv",
    "WRF-CESM/RCP8.5_DEWPT.csv",
    "WRF-CESM/RCP8.5_PRECIP.csv",
    "WRF-CESM/RCP8.5_SWDNB.csv",
    "WRF-CESM/RCP8.5_T2.csv",
    "WRF-CESM/RCP8.5_WSPD10.csv",
]

# Inspect shapefiles
print("\n=== Inspecting Shapefiles ===")
for shp in shapefiles:
    filepath = os.path.join(base_dir, shp)
    if os.path.exists(filepath):
        print(f"\nFile: {shp}")
        gdf = gpd.read_file(filepath)
        print("First 5 rows:")
        print(gdf.head())
        print("\nColumns:")
        print(gdf.columns)
        print(f"CRS (Coordinate Reference System): {gdf.crs}")
    else:
        print(f"File not found: {shp}")

# Inspect CSV files
print("\n=== Inspecting CSV Files ===")
for csv in csv_files:
    filepath = os.path.join(base_dir, csv)
    if os.path.exists(filepath):
        print(f"\nFile: {csv}")
        df = pd.read_csv(filepath)
        print("First 5 rows:")
        print(df.head())
        print("\nColumns:")
        print(df.columns)
        print(f"Number of rows: {len(df)}")
    else:
        print(f"File not found: {csv}")
