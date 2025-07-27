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
import seaborn as sns
import networkx as nx
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

def feature_selection(df, threshold=0.95):
    # Calculate correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop features with high correlation
    df_reduced = df.drop(columns=to_drop)

    # Return the reduced dataframe
    return df_reduced.columns.tolist()

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

# Apply feature selection based on the flag
apply_feature_selection = False  # Set this flag to True or False as needed
feature_selection_threshold = 0.85  # Set the threshold for feature selection
if apply_feature_selection:
    selected_features = feature_selection(df[numeric_cols], threshold=feature_selection_threshold)
    dropped_features = set(numeric_cols) - set(selected_features)
    print("\nDropped Features:")
    print(dropped_features)
    df = df[selected_features + ['Name']]  # Keep the 'Name' and 'huc12' columns
else:
    print("\nFeature selection not applied.")

# Update the list of numeric columns after feature selection
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
print("\nNumeric Columns (After Feature Selection):")
print(numeric_cols)
print("\nData shape after feature selection:")
print(df.shape)

# ------------------------------
# Correlation Matrix for Numeric Attributes
# ------------------------------
numeric_df = df[numeric_cols]
corr_matrix = numeric_df.corr()

# Ensure the correlation matrix contains only finite values (fill NaNs with 0 if any)
if not np.all(np.isfinite(corr_matrix)):
    corr_matrix = corr_matrix.fillna(0)
    print("Non-finite values found in the correlation matrix; filled NaNs with 0.")

# ------------------------------
# Correlation Heatmap for Numeric Attributes
# ------------------------------
# Convert the correlation matrix to a sparse CSR matrix for the algorithm
sparse_corr = csr_matrix(corr_matrix.values)
# Compute the permutation vector using Reverse Cuthill-McKee algorithm
perm = reverse_cuthill_mckee(sparse_corr, symmetric_mode=True)
# Reorder the correlation matrix using the permutation indices
reordered_corr = corr_matrix.iloc[perm, :].iloc[:, perm]

plt.figure(figsize=(36, 30))
sns.heatmap(reordered_corr, annot=True, cmap='coolwarm', fmt=".2f",
            xticklabels=reordered_corr.columns, yticklabels=reordered_corr.columns)
plt.title("Reordered Correlation Heatmap (Reverse Cuthill-McKee)")
plt.tight_layout()

# Save the heatmap figure to the imgs folder
heatmap_file = os.path.join(img_dir, "rcm_reordered_correlation_heatmap.png")
plt.savefig(heatmap_file)
plt.close()
print(f"Reordered correlation heatmap saved to {heatmap_file}")

# ------------------------------
# Correlation matrix graph for all numeric attributes
# ------------------------------
# Build the Graph with edges where |corr| > 0.75
threshold = 0.75
G = nx.Graph()

# Add all columns as nodes
for col in corr_matrix.columns:
    G.add_node(col)

# Add edges for high correlations
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > threshold:
            col_i = corr_matrix.columns[i]
            col_j = corr_matrix.columns[j]
            # Store correlation as an edge attribute called 'weight'
            G.add_edge(col_i, col_j, weight=corr_val)

# Build a subgraph containing only positive correlations so that strongly positive nodes cluster together.
G_pos = nx.Graph()
for u, v, d in G.edges(data=True):
    if d['weight'] > threshold:
        G_pos.add_edge(u, v, weight=d['weight'])
# Ensure all nodes are present
for node in G.nodes():
    if node not in G_pos:
        G_pos.add_node(node)

# Compute layout based on the positive subgraph.
# Increase 'k' to increase spacing between clusters.
pos = nx.spring_layout(G_pos, k=1.5, seed=42, iterations=200)

# --- Organize the Graph Layout Static ---
# Draw the Full Graph Using the Computed Positions ---
fig, ax = plt.subplots(figsize=(20, 20))
ax.set_title(f"Graph of Columns with |corr| > {threshold}\n(Strongly positive correlations clustered)", fontsize=18)

# Draw nodes with labels
nx.draw_networkx_nodes(G, pos, node_color='lightblue', edgecolors='black', node_size=600, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

# Draw edges.
# Edge colors will be based on the correlation values using the coolwarm colormap.
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
edges = nx.draw_networkx_edges(
    G, pos,
    edge_color=edge_weights,
    edge_cmap=plt.cm.coolwarm,
    edge_vmin=-1,
    edge_vmax=1,
    width=2,
    ax=ax
)

# Add a colorbar to indicate correlation values.
sm = mpl.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=-1, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.7, label='Correlation')

ax.axis('off')
graph_file = os.path.join(img_dir, "column_correlation_graph.png")
plt.savefig(graph_file, dpi=300, bbox_inches='tight')
plt.show()
print(f"Graph visualization saved to: {graph_file}")

# --- Organize the Graph Layout Interactive ---
edge_traces = []
# For each edge in the graph, create a separate trace (so that edge color can reflect the correlation)
for u, v, d in G.edges(data=True):
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    weight = d['weight']
    # Map the correlation value to a color using the coolwarm colormap
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    cmap = cm.get_cmap("coolwarm")
    color = mcolors.to_hex(cmap(norm(weight)))
    
    trace = go.Scatter(
        x=[x0, x1, None],
        y=[y0, y1, None],
        mode='lines',
        line=dict(width=2, color=color),
        hoverinfo='text',
        text=f"{u} - {v}: {weight:.2f}"
    )
    edge_traces.append(trace)

# Create node trace
node_x = []
node_y = []
node_text = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(node)

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode='markers+text',
    text=node_text,
    textposition="bottom center",
    hoverinfo='text',
    marker=dict(
        size=20,
        color='lightblue',
        line_width=2
    )
)

# Combine all traces into a figure
fig = go.Figure(
    data=edge_traces + [node_trace],
    layout=go.Layout(
        title=f"Interactive Graph of Columns with |corr| > {threshold}",
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        # Enable zoom and pan
        dragmode='zoom'
    )
)

# Save as Interactive HTML and Show
output_file = os.path.join(img_dir, "interactive_column_correlation_graph.html")
fig.write_html(output_file)
fig.show()
print(f"Interactive graph saved to: {output_file}")

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