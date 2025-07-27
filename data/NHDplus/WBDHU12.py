import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# File path to the WBDHU12 shapefile
shapefile_path = "/home/kumarv/xu000114/floods_droughts/NHDplus/WBD_07_HU2_Shape/Shape/WBDHU12.shp"

# Read the shapefile using geopandas
wbdhu12_gdf = gpd.read_file(shapefile_path)

# Print information about the WBDHU12 shapefile
print("WBDHU12 Shapefile Information:")
print(wbdhu12_gdf.info())
print("\nWBDHU12 Shapefile Head:")
print(wbdhu12_gdf.head())

# Load the CSV file
csv_file_path = os.path.join(os.getcwd(), "nhd_attributes_HUC0703.csv")
df = pd.read_csv(csv_file_path)
# this last column is all NA
df = df.iloc[:, :-1]

# Ensure that the 'huc12' values in the DataFrame have leading zeros to match the format in the GeoDataFrame
df['huc12'] = df['huc12'].apply(lambda x: str(x).zfill(12))

# Basic overview of the data
print("\nData shape:")
print(df.shape)
print("First few rows:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe())
print()

# Identify numeric columns for analysis
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
print("\nNumeric Columns:")
print(numeric_cols)
# Record the indices before dropping
initial_indices = set(df.index)

# Drop rows that have NA in any of the numeric columns
df_clean = df.dropna(subset=numeric_cols)
# Identify dropped rows by their indices
dropped_rows = sorted(list(initial_indices - set(df_clean.index)))
# Extract and print the indices and the corresponding "name" column values
dropped_info = df.loc[dropped_rows, 'Name']
print(f"\nDropped {len(dropped_info)} rows (indices) and their 'name' column values:")
print(dropped_info)
df = df_clean

# Identify constant columns (only one unique value)
constant_cols = [col for col in numeric_cols if df[col].nunique() == 1]
print(f"\nDropped {len(constant_cols)} constant columns:")
print(constant_cols)
# Drop these columns from the dataframe
df = df.drop(columns=constant_cols)

# Update the list of numeric columns after dropping constant columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
print("\nNumeric Columns (Updated):")
print(numeric_cols)
print("\nData shape after dropping rows and columns:")
print(df.shape)
print()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numeric_cols].values)

# K-means clustering
n_clusters = 5
colors = ['red', 'green', 'blue', 'purple', 'orange']
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)
df['cluster'] = cluster_labels  # add cluster labels to dataframe

# Apply t-SNE to reduce dimensions to 2 for visualization
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Create a DataFrame with t-SNE results and cluster labels for plotting
plot_df = pd.DataFrame({
    'TSNE1': X_tsne[:, 0],
    'TSNE2': X_tsne[:, 1],
    'cluster': cluster_labels.astype(str)  # Convert to string for categorical colors
})

# Match the huc12 from the CSV with the WBDHU12 shapefile
matched_gdf = wbdhu12_gdf[wbdhu12_gdf['huc12'].isin(df['huc12'])]

# Ensure the matched_gdf is not empty
if matched_gdf.empty:
    print("No matching huc12 found in the shapefile.")
    # Plot the shapefile
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    wbdhu12_gdf.plot(ax=ax, edgecolor='black', facecolor='lightblue')
    plt.title("WBDHU12 Shapefile Visualization", fontsize=14)
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    plt.grid(True)
    plt.savefig('/home/kumarv/xu000114/floods_droughts/NHDplus/imgs/WBDHU12_visualization.png')
    plt.show()
else:
    # Plot the shapefile with matched locations marked in blue
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    wbdhu12_gdf.plot(ax=ax, edgecolor='black', facecolor='lightblue', alpha=0.5)
    matched_gdf.plot(ax=ax, edgecolor='red', facecolor='red', alpha=0.5)
    plt.title("WBDHU12 Shapefile Visualization with Matched Locations", fontsize=14)
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    plt.grid(True)
    plt.savefig('/home/kumarv/xu000114/floods_droughts/NHDplus/imgs/WBDHU12_visualization_matched.png')
    plt.show()

    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    matched_gdf.plot(ax=ax, edgecolor='black', facecolor='lightblue', alpha=0.5)
    plt.title("WBDHU12 Matched Locations", fontsize=14)
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    plt.grid(True)
    plt.savefig('/home/kumarv/xu000114/floods_droughts/NHDplus/imgs/WBDHU12_visualization_matched_local.png')
    plt.show()

    # Plot the clusters on the map
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    matched_gdf.plot(ax=ax, edgecolor='black', facecolor='lightblue', alpha=0.5)
    for cluster in range(n_clusters):
        cluster_gdf = matched_gdf[matched_gdf['huc12'].isin(df[df['cluster'] == cluster]['huc12'])]
        cluster_gdf.plot(ax=ax, edgecolor=colors[cluster], alpha=0.5, label=f'Cluster {cluster}')
    plt.title("WBDHU12 Clusters Visualization", fontsize=14)
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/kumarv/xu000114/floods_droughts/NHDplus/imgs/WBDHU12_visualization_matched_local_cluster.png')
    plt.show()

# File path to the USA boundary shapefile
countries_shp = "/home/kumarv/xu000114/floods_droughts/eddev1/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"
world = gpd.read_file(countries_shp)

# Filter for mainland USA (excluding Alaska and Hawaii)
usa = world[(world['ADMIN'] == 'United States of America') & 
            (world['SUBUNIT'] != 'Alaska') & 
            (world['SUBUNIT'] != 'Hawaii')]

# Reproject WBDHU12 to WGS84
wbdhu12_gdf_wgs84 = wbdhu12_gdf.to_crs(epsg=4326)

if matched_gdf.empty:
    print("No matching huc12 found in the shapefile.")
    # Plot the shapefile with USA boundary
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    usa.boundary.plot(ax=ax, color='black', linewidth=1, label='USA Boundary')
    wbdhu12_gdf_wgs84.plot(ax=ax, edgecolor='blue', facecolor='lightblue', alpha=0.5, label='WBDHU12')
    plt.title("WBDHU12 Shapefile Visualization with USA Boundary", fontsize=14)
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig('/home/kumarv/xu000114/floods_droughts/NHDplus/imgs/WBDHU12_visualization_USA.png')
    plt.show()
else:
    # Plot the shapefile with USA boundary and matched locations marked in red
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    usa.boundary.plot(ax=ax, color='black', linewidth=1, label='USA Boundary')
    wbdhu12_gdf_wgs84.plot(ax=ax, edgecolor='blue', facecolor='lightblue', alpha=0.5, label='WBDHU12')
    matched_gdf.to_crs(epsg=4326).plot(ax=ax, edgecolor='red', facecolor='red', alpha=0.5, label='Matched Locations')
    plt.title("WBDHU12 Shapefile Visualization with USA Boundary and Matched Locations", fontsize=14)
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig('/home/kumarv/xu000114/floods_droughts/NHDplus/imgs/WBDHU12_visualization_USA_matched.png')
    plt.show()
