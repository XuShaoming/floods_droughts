# Climate Data Visualization Scripts

## Script Descriptions

### `inspect_data.py`
1. Check the values of the data.

### `shp_vis.py`
1. Visualize climate data polygons with USA contours.
2. Visualize climate data locations with USA contours.
3. Visualize the Kettle River Watershed at HUC8 and HUC12 levels.

### `generate_gif_1KNN.py`
1. Create GIF videos for the time period you define in the script.
2. Use the 1-KNN method to assign weather values to each HUC12 region.

### `generate_img_KNN_IDW.py`
1. Create an image of the selected weather variable at a given time for each HUC12 region.
2. Use KNN and Inverse distance weighting to assign weather values to each HUC12 region.

### `generate_gif_KNN_IDW.py`
1. Create GIF videos for the time period you define in the script.
2. Use KNN and Inverse distance weighting to assign weather values to each HUC12 region.

## Machine Learning Modeling

The Machine Learning (ML) team processes climate data and watershed information into appropriate formats and scales for ML modeling. Specifically, datasets have been processed to the HUC8 level at hourly resolution.

### Data Sources and Resolution
- **EDDEV1 Weather Data**: 36km × 36km resolution covering the continental United States
- **HSPF Simulation Data**: Includes streamflow at HUC8 level in hourly resolution
- **Watershed Shapes**: Available at HUC12 level

### Processing Pipeline

#### 1. Weather Point Selection
Since EDDEV1 weather data covers the entire continental US, the ML team implemented a K-Nearest Neighbor (KNN) algorithm to automatically identify relevant weather data points near a given watershed. These points, at 36km × 36km resolution, are often coarser than the HUC12 watershed boundaries. For example, the Kettle River watershed contains dozens of HUC12 sub-watersheds but is only covered by 9 weather points.

#### 2. Spatial Downscaling
To address the resolution mismatch, a downscaling algorithm using Inverse Distance Weighting (IDW) interpolates weather observations from the coarse grid to each HUC12 sub-watershed. This provides higher spatial resolution that better matches the watershed boundaries.

#### 3. Aggregation to HUC8
After obtaining weather data for each HUC12 sub-watershed, an area-weighted aggregation algorithm combines these values to produce weather observations at the HUC8 level.

#### 4. Time Series Generation
The HUC8 weather observations are merged with HSPF streamflow observations based on timestamps, creating comprehensive hourly time series datasets for Minnesota watersheds.

All processing steps are implemented in Python and can be automatically applied to generate consistent datasets for any watershed in Minnesota.
