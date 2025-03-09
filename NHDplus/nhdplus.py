import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go

# Load the CSV file
img_dir = os.path.join(os.getcwd(), "imgs")
file_path = os.path.join(os.getcwd(), "nhd_attributes_HUC0703.csv")
df = pd.read_csv(file_path)
# this last column is all NA
df = df.iloc[:, :-1]

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
    'cluster': cluster_labels.astype(str),  # Convert to string for categorical colors
    'hover': df['Name'].astype(str)  # Use 'Name' column as hover text
})

# Sort the DataFrame by cluster to ensure consistent coloring
plot_df = plot_df.sort_values(by='cluster')

# Create an interactive scatter plot with Plotly Express
fig = px.scatter(
    plot_df, 
    x='TSNE1', 
    y='TSNE2', 
    color='cluster',
    color_discrete_sequence=colors,  # Use the same colors as the map
    custom_data=['hover'],  # include hover text as custom data
    hover_data={'hover': True, 'TSNE1': False, 'TSNE2': False},
    title=f"t-SNE Clustering (n_clusters = {n_clusters})",
    labels={'TSNE1': 't-SNE Dimension 1', 'TSNE2': 't-SNE Dimension 2', 'cluster': 'Cluster'}
)

# Enhance layout for better visibility and interactive exploration
fig.update_layout(
    width=800,
    height=800,
    xaxis_title="t-SNE Dimension 1",
    yaxis_title="t-SNE Dimension 2",
    legend_title="Cluster"
)

# Now, because Plotly Express groups points into separate traces (one per cluster),
# we need to update each trace with the correct text values.
# We'll extract each trace’s customdata (which contains the correct "hover" values for that trace).
trace_indices = list(range(len(fig.data)))
show_text_all = []
hide_text_all = []
for trace in fig.data:
    # trace.customdata is a 2D array with one column (since we passed ['hover'])
    # We extract that column as a list for this trace.
    show_text_all.append(trace.customdata[:, 0].tolist())
    hide_text_all.append([""] * len(trace.customdata))

# Add interactive buttons to toggle point names.
# The update menu will update every trace (using trace_indices) with its corresponding text.
fig.update_layout(
    updatemenus=[
        {
            "type": "buttons",
            "direction": "left",
            "buttons": [
                {
                    "label": "Show Names",
                    "method": "restyle",
                    "args": [
                        {"mode": "markers+text", "text": show_text_all, "hoverinfo": "text"},
                        trace_indices
                    ],
                },
                {
                    "label": "Hide Names",
                    "method": "restyle",
                    "args": [
                        {"mode": "markers", "text": hide_text_all, "hoverinfo": "text"},
                        trace_indices
                    ],
                },
            ],
            "pad": {"r": 10, "t": 10},
            "showactive": True,
            "x": 0.0,
            "xanchor": "left",
            "y": 1.1,
            "yanchor": "top",
        }
    ]
)

# Save the interactive plot as an HTML file
output_file = os.path.join(img_dir, "interactive_basin_clustering_tsne.html")
fig.write_html(output_file)
fig.show()

print(f"Interactive t-SNE clustering visualization saved to: {output_file}")

# File path to the WBDHU12 shapefile
shapefile_path = "/home/kumarv/xu000114/floods_droughts/NHDplus/WBD_07_HU2_Shape/Shape/WBDHU12.shp"

# Read the shapefile using geopandas
wbdhu12_gdf = gpd.read_file(shapefile_path)

# Ensure that the 'huc12' values in the DataFrame have leading zeros to match the format in the GeoDataFrame
df['huc12'] = df['huc12'].apply(lambda x: str(x).zfill(12))

# Match the huc12 from the CSV with the WBDHU12 shapefile
matched_gdf = wbdhu12_gdf[wbdhu12_gdf['huc12'].isin(df['huc12'])]

# Plot the clusters on the map
plt.figure(figsize=(10, 10))
ax = plt.gca()
matched_gdf.plot(ax=ax, edgecolor='black', facecolor='lightblue', alpha=0.5)
for cluster in range(n_clusters):
    cluster_gdf = matched_gdf[matched_gdf['huc12'].isin(df[df['cluster'] == cluster]['huc12'])]
    cluster_gdf.plot(ax=ax, edgecolor='black', facecolor=colors[cluster], alpha=0.5, label=f'Cluster {cluster}')
plt.title("WBDHU12 Clusters Visualization", fontsize=14)
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig('/home/kumarv/xu000114/floods_droughts/NHDplus/imgs/WBDHU12_visualization_matched_local_cluster.png')
plt.show()

# Create an interactive map with Plotly
fig_map = go.Figure()

# Add the WBDHU12 boundaries
fig_map.add_trace(go.Scattergeo(
    lon=matched_gdf.geometry.centroid.x,
    lat=matched_gdf.geometry.centroid.y,
    mode='markers',
    marker=dict(size=2, color='lightblue'),
    name='WBDHU12'
))

# Add clusters
for cluster in range(n_clusters):
    cluster_gdf = matched_gdf[matched_gdf['huc12'].isin(df[df['cluster'] == cluster]['huc12'])]
    cluster_gdf = cluster_gdf.merge(df[['huc12', 'Name']], on='huc12')  # Merge to get 'Name' column
    fig_map.add_trace(go.Scattergeo(
        lon=cluster_gdf.geometry.centroid.x,
        lat=cluster_gdf.geometry.centroid.y,
        mode='markers+text',
        marker=dict(size=5, color=colors[cluster]),
        text=cluster_gdf['Name'],  # Use 'Name' column for hover text
        name=f'Cluster {cluster}',
        hoverinfo='text'
    ))

# Update layout for the map
fig_map.update_layout(
    title="WBDHU12 Clusters Visualization",
    geo=dict(
        scope='usa',
        projection_type='albers usa',
        showland=True,
        center=dict(lat=matched_gdf.geometry.centroid.y.mean(), lon=matched_gdf.geometry.centroid.x.mean()),  # Center the map
        fitbounds="locations"  # Fit the map to the locations
    ),
    legend_title="Cluster"
)

# Add interactive buttons to toggle point names
trace_indices_map = list(range(len(fig_map.data)))
show_text_all_map = []
hide_text_all_map = []
for trace in fig_map.data:
    if trace.text is not None:
        show_text_all_map.append(trace.text)
        hide_text_all_map.append([""] * len(trace.text))
    else:
        show_text_all_map.append([])
        hide_text_all_map.append([])

fig_map.update_layout(
    updatemenus=[
        {
            "type": "buttons",
            "direction": "left",
            "buttons": [
                {
                    "label": "Show Names",
                    "method": "restyle",
                    "args": [
                        {"mode": "markers+text", "text": show_text_all_map, "hoverinfo": "text"},
                        trace_indices_map
                    ],
                },
                {
                    "label": "Hide Names",
                    "method": "restyle",
                    "args": [
                        {"mode": "markers", "text": hide_text_all_map, "hoverinfo": "text"},
                        trace_indices_map
                    ],
                },
            ],
            "pad": {"r": 10, "t": 10},
            "showactive": True,
            "x": 0.0,
            "xanchor": "left",
            "y": 1.1,
            "yanchor": "top",
        }
    ]
)

# Save the interactive map as an HTML file
output_file_map = os.path.join(img_dir, "WBDHU12_visualization_matched_local_cluster.html")
fig_map.write_html(output_file_map)
fig_map.show()

print(f"Interactive WBDHU12 clusters visualization saved to: {output_file_map}")