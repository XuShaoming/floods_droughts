# Flood and Drought Prediction with Deep Learning

This project implements an end-to-end pipeline for flood and drought prediction using neural networks. It processes meteorological data (EDDEV1) and streamflow data (HSPF) to create machine learning models for watershed management and hydrological forecasting.

## Project Structure

### Configuration and Utility Files

#### `config.yaml` - Master Configuration File
Central configuration using YAML anchors and inheritance:
- **Base configuration**: Shared settings across all experiments (`&BASE_CONFIG`)
- **Inheritance system**: Experiments inherit base settings and override specific parameters
- **Available experiments**: `hourly_flood_events`, `multitarget`, `streamflow_exp1`
- **Documentation**: Each experiment includes detailed hydrological rationale
- **Maintainability**: Easy to add new experiments and modify existing ones

#### Shell Scripts
- **`aga36.sh`** and **`aga37.sh`**: Batch processing scripts for cluster computing environments


### Core Machine Learning Components

#### `dataloader.py` - Data Processing Engine
Comprehensive data loading and preprocessing utilities for time series modeling:
- **Sliding window generation**: Creates overlapping time series windows with configurable stride
- **Multi-target support**: Handles multiple prediction targets simultaneously
- **Data normalization**: StandardScaler integration with proper train/test splitting
- **Memory-efficient processing**: Optimized for large datasets with minimal memory usage
- **PyTorch integration**: Native Dataset and DataLoader support

#### `train.py` - Training
Main training script with YAML configuration support:
- **YAML configuration**: Centralized experiment management with inheritance
- **Model training**: Supports LSTM and other neural network architectures
- **Early stopping**: Prevents overfitting with configurable patience
- **Learning rate scheduling**: Adaptive learning rate adjustment during training
- **TensorBoard logging**: Real-time training metrics and visualizations
- **Checkpointing**: Saves best models and training states


#### `inference.py` - Model Evaluation System
Complete model evaluation and analysis pipeline:
- **Model loading**: Automatic model and configuration loading from experiment directories
- **Comprehensive metrics**: MSE, RMSE, MAE, R², NSE, KGE calculations
- **Rich visualizations**: Scatter plots, time series reconstruction, residual analysis
- **Denormalization**: Proper scaling back to original units using saved scalers
- **Results export**: Automated saving of predictions and analysis results

#### `models/LSTMModel.py` - Neural Network Architecture
LSTM model implementation optimized for hydrological time series:
- **Configurable architecture**: Variable layers, hidden sizes, and dropout rates
- **Batch normalization**: Improved training stability and convergence
- **Multi-task outputs**: Support for predicting multiple variables simultaneously
- **Proper initialization**: Xavier/Glorot initialization for stable training

### Data Processing Pipeline

#### `process_eddev_data.py` - Meteorological Data Processor
Processes EDDEV1 climate data for watershed analysis using spatial interpolation:
- **Variables processed**: Temperature (T2), Dew Point (DEWPT), Precipitation (PRECIP), Solar Radiation (SWDNB), Wind Speed (WSPD10), Latent Heat (LH)
- **Spatial methods**: K-Nearest Neighbors (KNN) + Inverse Distance Weighting (IDW)
- **Temporal resolution**: Hourly data processing with timezone handling
- **Geographic projection**: Proper coordinate system handling for accurate distance calculations
- **Batch processing**: Supports date range processing and multiple climate scenarios

```bash
# Usage examples
python process_eddev_data.py --all --basin "KettleR_Watersheds" --scenario "Historical"
```

#### `process_flow_data.py` - Streamflow Data Processor
Merges and processes HSPF simulation data:
- **Data integration**: Combines hourly flow data with daily output metrics
- **Temporal interpolation**: Interpolates daily values to hourly resolution
- **Multiple scenarios**: Supports historical, RCP4.5, and RCP8.5 climate scenarios
- **Basin support**: Handles multiple watershed basins (KettleRiverModels, BlueEarth, LeSueur)
```bash
# Usage examples
python process_flow_data.py
```

#### `combine_eddev_flow.py` - Data Integration Engine
Combines meteorological and streamflow data into analysis-ready datasets:
- **Datetime alignment**: Precise temporal matching between weather and flow data
- **Data validation**: Automatic file detection and format verification
- **Scenario matching**: Handles different naming conventions between datasets
- **Output formatting**: Creates standardized CSV files for machine learning

```bash
# Usage examples
python combine_eddev_flow.py --basin "KettleRiverModels" --scenario "hist_scaled"
```

### Visualization and Analysis Tools

#### `combined_visualize.py` - Integrated Data Visualization
Creates comprehensive plots combining meteorological and streamflow data:
- **Multi-variable plotting**: Streamflow vs. precipitation relationships
- **Time series analysis**: Shows hydrological responses to weather events
- **Custom date ranges**: Focuses on specific events or periods
- **Publication outputs**: High-quality figures for reports and papers


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

## LSTM Training for Flood/Drought Prediction

### Overview
The project includes an LSTM-based deep learning pipeline for flood and drought prediction using time series data. The training system uses YAML configuration files with inheritance for better maintainability and reproducibility.

### Key Files
- **`train.py`**: Main training script with YAML configuration support
- **`config.yaml`**: Single configuration file with base config and all experiments using inheritance
- **`dataloader.py`**: Comprehensive data loading and preprocessing utilities
- **`inference.py`**: Model inference and evaluation script

### Quick Start
```bash
# Use specific hourly streamflow experiments
python train.py --config 'config.yaml' --experiment streamflow_exp1 --seed 42
# Evaluate the trained model
python inference.py --model-dir experiments/streamflow_exp1 --model-trained best_model.pth --dataset test --analysis
```

### Features
- YAML-based configuration management
- Many-to-many and many-to-one sequence modeling
- Multi-target prediction support
- Sliding window time series processing
- Early stopping and learning rate scheduling
- TensorBoard logging and visualization
- Comprehensive model evaluation metrics
- Time series reconstruction and validation

### Advanced Analysis Tools

<!-- #### `NHDplus/nhdplus.py` - Watershed Attribute Analysis
Advanced analysis of National Hydrography Dataset Plus (NHDplus) attributes:
- **Feature selection**: Correlation-based feature reduction for watershed characteristics
- **Clustering analysis**: K-means clustering of watershed attributes
- **Dimensionality reduction**: t-SNE visualization of watershed similarity
- **Network analysis**: Stream network topology and connectivity analysis
- **Visualization**: Interactive plots using Plotly for watershed exploration

#### `NHDplus/WBDHU12.py` - HUC12 Watershed Processing
Specialized processing for HUC12 level watersheds:
- **Boundary processing**: Watershed delineation and geometric operations
- **Attribute extraction**: Physical and hydrological characteristics
- **Multi-scale analysis**: Links between HUC8 and HUC12 scales

#### `NHDplus/nhdplus_attributes.py` - Attribute Processing
Processing and analysis of watershed physical characteristics from NHDplus dataset

## YAML Configuration System

The project uses a sophisticated YAML configuration system with inheritance for maintainable experiment management. -->

### `config.yaml` - Master Configuration File
Central configuration using YAML anchors and inheritance:
- **Base configuration**: Shared settings across all experiments (`&BASE_CONFIG`)
- **Inheritance system**: Experiments inherit base settings and override specific parameters
- **Documentation**: Each experiment includes detailed hydrological rationale
- **Maintainability**: Easy to add new experiments and modify existing ones

### 1. Data Processing Pipeline

```bash
# Process meteorological data (EDDEV1)
python process_eddev_data.py --all --basin "KettleR_Watersheds" --scenario "Historical"

# Process streamflow data (HSPF)
python process_flow_data.py --basin "KettleRiverModels" --scenario "hist_scaled"

# Combine datasets
python combine_eddev_flow.py --basin "KettleRiverModels" --scenario "hist_scaled"
```


## Installation and Dependencies

### Required Python Packages

```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install pandas numpy matplotlib seaborn
pip install scikit-learn scipy
pip install geopandas shapely fiona
pip install pyproj cartopy
pip install tqdm pyyaml tensorboard

# Optional for advanced analysis
pip install plotly networkx
pip install jupyter notebook
```

### Data Requirements

The project expects the following data structure:

```
floods_droughts/
├── eddev1/                          # EDDEV1 meteorological data
│   ├── WRF-CESM/                   # Weather data files
│   ├── Climate_Data_Locations.*     # Weather station coordinates
│   ├── KettleR_Watersheds_NewMetSeg.* # Watershed shapefiles
│   └── Lat_Lon_Centroid_Locations.csv
├── flow_data/                       # HSPF streamflow simulations
│   ├── *_FLOW.csv                  # Hourly streamflow data
│   └── *_Daily_outputs.csv         # Daily metrics
└── processed/                       # Output directory for processed data
```
