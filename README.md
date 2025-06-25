# Flood and Drought Prediction with Deep Learning

This project implements an end-to-end pipeline for flood and drought prediction using LSTM neural networks. It processes meteorological data (EDDEV1) and streamflow data (HSPF) to create machine learning models for watershed management and hydrological forecasting.

## Project Structure

### Core Machine Learning Components

#### `train.py` - LSTM Training Engine
Main training script for LSTM-based flood/drought prediction with comprehensive YAML configuration support:
- **Multi-task learning**: Support for multiple target variables (streamflow, evapotranspiration)
- **Many-to-many sequence modeling**: Time series prediction with configurable window sizes
- **Experiment management**: YAML-based configuration with inheritance for reproducible research
- **GPU acceleration**: Automatic device detection and optimization
- **Advanced training**: Early stopping, learning rate scheduling, gradient clipping
- **Logging**: TensorBoard integration and comprehensive metrics tracking

```bash
# Quick start examples
python train.py                                    # Default experiment
python train.py --experiment hourly_short_term     # Flash flood prediction
python train.py --experiment hourly_flood_events   # Multi-day flood analysis
python train.py --override training.epochs=50      # Runtime parameter override
```

#### `dataloader.py` - Data Processing Engine
Comprehensive data loading and preprocessing utilities for time series modeling:
- **Sliding window generation**: Creates overlapping time series windows with configurable stride
- **Multi-target support**: Handles multiple prediction targets simultaneously
- **Data normalization**: StandardScaler integration with proper train/test splitting
- **Memory-efficient processing**: Optimized for large datasets with minimal memory usage
- **PyTorch integration**: Native Dataset and DataLoader support

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
python process_eddev_data.py --start "1975-01-01" --end "1975-01-31" --basin "KettleR_Watersheds" --scenario "Historical"
python process_eddev_data.py --all --basin "KettleR_Watersheds" --scenario "RCP4.5"
```

#### `process_flow_data.py` - Streamflow Data Processor
Merges and processes HSPF simulation data:
- **Data integration**: Combines hourly flow data with daily output metrics
- **Temporal interpolation**: Interpolates daily values to hourly resolution
- **Multiple scenarios**: Supports historical, RCP4.5, and RCP8.5 climate scenarios
- **Basin support**: Handles multiple watershed basins (KettleRiverModels, BlueEarth, LeSueur)

#### `combine_eddev_flow.py` - Data Integration Engine
Combines meteorological and streamflow data into analysis-ready datasets:
- **Datetime alignment**: Precise temporal matching between weather and flow data
- **Data validation**: Automatic file detection and format verification
- **Scenario matching**: Handles different naming conventions between datasets
- **Output formatting**: Creates standardized CSV files for machine learning

### Visualization and Analysis Tools

#### `shp_vis.py` - Geospatial Visualization
Creates high-quality maps for spatial data analysis:
- **Watershed visualization**: HUC8 and HUC12 level watershed boundaries
- **Climate grid display**: Weather station locations and data coverage
- **Geographic context**: USA contours and coordinate system visualization
- **Publication-ready outputs**: High-resolution PNG exports for reports

#### `generate_gif_1KNN.py` - Temporal Animation (1-KNN Method)
Creates animated visualizations of meteorological variables over time:
- **Method**: 1-Nearest Neighbor assignment for each watershed
- **Variables**: All EDDEV1 meteorological parameters
- **Output**: Time-lapse GIF animations showing spatial-temporal patterns
- **Customizable**: User-defined time periods and visualization parameters

#### `generate_img_KNN_IDW.py` - Static Spatial Maps (KNN-IDW Method)
Generates static images of meteorological variables at specific times:
- **Method**: K-Nearest Neighbors with Inverse Distance Weighting
- **Spatial interpolation**: Higher accuracy than 1-KNN method
- **Single timestep**: Detailed analysis of specific weather events
- **Multiple variables**: Temperature, precipitation, radiation, wind patterns

#### `generate_gif_KNN_IDW.py` - Advanced Temporal Animation
Creates sophisticated animated visualizations using KNN-IDW interpolation:
- **Enhanced accuracy**: Combines multiple weather stations for better spatial representation
- **Smooth interpolation**: IDW creates realistic spatial gradients
- **Time series animations**: Shows evolution of weather patterns over time
- **Research quality**: Suitable for scientific presentations and publications

### Data Inspection and Quality Control

#### `inspect_data.py` - Data Quality Assessment
Comprehensive data validation and inspection tool:
- **Shapefile validation**: Checks watershed geometry and attributes
- **Centroid verification**: Validates weather station coordinates
- **Data coverage**: Analyzes temporal and spatial data availability
- **Format consistency**: Ensures proper data types and structures

#### `flow_data_inspect.py` - Streamflow Data Analysis
Detailed analysis of HSPF simulation outputs:
- **Statistical summaries**: Min, max, mean, median, standard deviation
- **Time series plotting**: Visual inspection of flow patterns
- **Data quality checks**: Identifies gaps, outliers, and anomalies
- **Multiple scenarios**: Compares historical vs. future climate projections

#### `combined_visualize.py` - Integrated Data Visualization
Creates comprehensive plots combining meteorological and streamflow data:
- **Multi-variable plotting**: Streamflow vs. precipitation relationships
- **Time series analysis**: Shows hydrological responses to weather events
- **Custom date ranges**: Focuses on specific events or periods
- **Publication outputs**: High-quality figures for reports and papers

### Utility and Example Scripts

#### `run_example.py` - Getting Started Guide
Demonstrates complete workflow from training to inference:
- **Step-by-step examples**: Shows how to use all major components
- **Error handling**: Proper error checking and user guidance
- **Configuration examples**: Demonstrates different experiment setups
- **Best practices**: Shows recommended usage patterns

#### `demo_inheritance.py` - Configuration Tutorial
Explains YAML inheritance system for experiment management:
- **Configuration examples**: Shows how YAML anchors and inheritance work
- **Experiment comparison**: Demonstrates differences between experiments
- **Best practices**: Proper configuration file organization
- **Documentation**: Helps users create their own experiments

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
# Use default experiment (hourly flood events)
python train.py

# Use specific hourly streamflow experiments
python train.py --experiment hourly_short_term    # 3-day window, hourly stride
python train.py --experiment hourly_daily_cycles  # 1-week window, 6-hour stride  
python train.py --experiment hourly_flood_events  # 5-day window, 12-hour stride
python train.py --experiment hourly_baseflow      # 15-day window, daily stride
python train.py --experiment multitarget         # Multi-task: streamflow + ET

# Override specific values at runtime
python train.py --experiment hourly_short_term --override training.epochs=50 data.window_size=96
```

### Available Experiments
| Experiment | Window | Stride | Purpose | Memory | Time |
|------------|--------|--------|---------|--------|------|
| `hourly_short_term` | 72h (3d) | 1h | Flash floods, immediate response | Low | ~30-60min |
| `hourly_daily_cycles` | 168h (7d) | 6h | Daily patterns, snowmelt cycles | Medium | ~1-2h |
| `hourly_flood_events` | 120h (5d) | 12h | Multi-day flood event analysis | Med-High | ~2-3h |
| `hourly_baseflow` | 360h (15d) | 24h | Seasonal trends, drought monitoring | High | ~3-6h |
| `multitarget` | 30h | 1h | Joint streamflow + ET prediction | Low-Med | ~1-2h |

### YAML Configuration with Inheritance
The configuration uses YAML anchors (`&`) and inheritance (`<<: *`) for maintainable experiment management:

```yaml
# Base configuration shared by all experiments
base_config: &BASE_CONFIG
  data:
    csv_file: "processed/data.csv"
    window_size: 30
    target_cols: ["streamflow"]
  model:
    hidden_size: 64
    num_layers: 2

# Experiments inherit from base and override specific values
hourly_short_term:
  <<: *BASE_CONFIG
  data:
    window_size: 72    # Override: 3 days for storm events
    stride: 1          # Override: hourly resolution
  model:
    hidden_size: 64    # Keep base value
```

### Best Practices
- **Experiment Naming**: Use descriptive names like `hourly_urban_floods`, `seasonal_drought_prediction`
- **Documentation**: Each experiment in `config.yaml` includes detailed hydrological rationale
- **Memory Management**: Use smaller batch sizes for longer window experiments
- **Validation**: Test on different years/seasons and extreme events

### Configuration Structure
```yaml
data:           # Data loading and preprocessing
  csv_file: "path/to/data.csv"
  window_size: 30
  target_cols: ["streamflow"]
  train_years: [1980, 2000]
  
model:          # LSTM architecture
  hidden_size: 64
  num_layers: 2
  dropout: 0.2
  
training:       # Training hyperparameters
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  scheduler:
    type: "ReduceLROnPlateau"
    patience: 5
    
output:         # Logging and saving
  save_dir: "models"
  tensorboard_log: true
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

#### `NHDplus/nhdplus.py` - Watershed Attribute Analysis
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

## YAML Configuration System

The project uses a sophisticated YAML configuration system with inheritance for maintainable experiment management.

### `config.yaml` - Master Configuration File
Central configuration using YAML anchors and inheritance:
- **Base configuration**: Shared settings across all experiments (`&BASE_CONFIG`)
- **Inheritance system**: Experiments inherit base settings and override specific parameters
- **Documentation**: Each experiment includes detailed hydrological rationale
- **Maintainability**: Easy to add new experiments and modify existing ones

### Available Experiments

| Experiment | Window | Stride | Purpose | Memory | Training Time |
|------------|--------|--------|---------|--------|---------------|
| `hourly_short_term` | 72h (3 days) | 1h | Flash floods, immediate response | Low | ~30-60min |
| `hourly_daily_cycles` | 168h (7 days) | 6h | Daily patterns, snowmelt cycles | Medium | ~1-2h |
| `hourly_flood_events` | 120h (5 days) | 12h | Multi-day flood event analysis | Med-High | ~2-3h |
| `hourly_baseflow` | 360h (15 days) | 24h | Seasonal trends, drought monitoring | High | ~3-6h |
| `multitarget` | 30h | 1h | Joint streamflow + ET prediction | Low-Med | ~1-2h |

### Experiment Details

#### `hourly_short_term`
- **Purpose**: Flash flood prediction (1-12h ahead)
- **Hydrological rationale**: Captures immediate runoff response in urban watersheds
- **Features**: T2, DEWPT, PRECIP, SWDNB, WSPD10, LH
- **Model**: 64 hidden units, 2 layers
- **Optimal for**: Urban watersheds, rapid response systems

#### `hourly_daily_cycles`
- **Purpose**: Daily cycle modeling (snowmelt, ET patterns)
- **Hydrological rationale**: One week captures multiple daily cycles and weekly patterns
- **Features**: All available meteorological variables
- **Model**: 96 hidden units, 3 layers
- **Optimal for**: Natural watersheds, snowmelt-dominated systems

#### `hourly_flood_events`
- **Purpose**: Multi-day flood event analysis
- **Hydrological rationale**: Captures full flood lifecycle (rise, peak, recession)
- **Features**: PRECIP, T2, SWDNB, WSPD10
- **Model**: 128 hidden units, 3 layers
- **Optimal for**: River basin management, flood forecasting

#### `hourly_baseflow`
- **Purpose**: Baseflow and seasonal trends
- **Hydrological rationale**: 15 days captures seasonal baseflow trends and groundwater response
- **Features**: T2, DEWPT, ET, SWDNB (energy balance variables)
- **Model**: 96 hidden units, 2 layers
- **Optimal for**: Water supply planning, drought monitoring

#### `multitarget`
- **Purpose**: Multi-task learning (Streamflow + Evapotranspiration)
- **Hydrological rationale**: Joint prediction of water/energy fluxes for water balance
- **Features**: Complete meteorological suite
- **Model**: 96 hidden units, 2 layers, 2 output targets
- **Optimal for**: Water balance studies, ecosystem modeling

### Configuration Structure

```yaml
# Example experiment configuration
experiment_name:
  <<: *BASE_CONFIG          # Inherit from base
  data:
    window_size: 120         # Override: 5-day window
    stride: 12               # Override: 12-hour stride
    feature_cols: ["T2", "PRECIP", "SWDNB"]  # Override: specific features
  model:
    hidden_size: 128         # Override: larger model
    num_layers: 3           # Override: deeper network
  training:
    batch_size: 16          # Override: smaller batches for memory
    epochs: 150             # Override: longer training
```

## Quick Start Guide

### 1. Data Processing Pipeline

```bash
# Process meteorological data (EDDEV1)
python process_eddev_data.py --all --basin "KettleR_Watersheds" --scenario "Historical"

# Process streamflow data (HSPF)
python process_flow_data.py --basin "KettleRiverModels" --scenario "hist_scaled"

# Combine datasets
python combine_eddev_flow.py --basin "KettleRiverModels" --scenario "hist_scaled"
```

### 2. Data Inspection and Visualization

```bash
# Inspect data quality
python inspect_data.py
python flow_data_inspect.py

# Create visualizations
python shp_vis.py                    # Spatial maps
python combined_visualize.py         # Time series plots
python generate_img_KNN_IDW.py       # Weather maps
```

### 3. Machine Learning Training

```bash
# Quick training with default settings
python train.py

# Specific experiments for different use cases
python train.py --experiment hourly_short_term    # Flash floods
python train.py --experiment hourly_flood_events  # Multi-day events
python train.py --experiment hourly_baseflow      # Drought monitoring

# Custom training with parameter overrides
python train.py --experiment hourly_short_term --override training.epochs=50 model.hidden_size=128
```

### 4. Model Evaluation

```bash
# Comprehensive model evaluation
python inference.py --model-dir experiments/hourly_flood_events --dataset test --analysis

# Evaluate specific datasets
python inference.py --model-dir experiments/hourly_short_term --dataset validation
```

### 5. Example Workflow

```bash
# Run complete example workflow
python run_example.py

# Demonstrate YAML inheritance
python demo_inheritance.py
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

## Performance and Hardware Requirements

### Memory Requirements
- **hourly_short_term**: 2-4 GB RAM
- **hourly_daily_cycles**: 4-8 GB RAM  
- **hourly_flood_events**: 6-12 GB RAM
- **hourly_baseflow**: 8-16 GB RAM

### GPU Support
- Automatic GPU detection and utilization
- Training time reduced by 3-5x with GPU
- Recommended: NVIDIA GPU with 4+ GB VRAM

### Storage Requirements
- Raw data: ~5-10 GB per watershed and scenario
- Processed data: ~1-2 GB per combined dataset
- Model outputs: ~100-500 MB per experiment

## Project Applications

### Flood Forecasting
- **Flash flood prediction**: `hourly_short_term` experiment
- **River flood forecasting**: `hourly_flood_events` experiment
- **Real-time warnings**: Integration with meteorological forecasts

### Drought Monitoring
- **Baseflow trends**: `hourly_baseflow` experiment
- **Seasonal predictions**: Long-term water availability
- **Agricultural planning**: ET and water balance modeling

### Water Resource Management
- **Reservoir operations**: Multi-day flood predictions
- **Supply planning**: Baseflow and seasonal trend analysis
- **Environmental flows**: Minimum flow requirements

### Climate Change Impact Assessment
- **Scenario analysis**: RCP4.5 and RCP8.5 projections
- **Extreme event changes**: Flood and drought frequency
- **Adaptation planning**: Infrastructure and policy decisions

## Research and Development

### Model Architecture
- **LSTM networks**: Optimized for hydrological time series
- **Multi-task learning**: Joint prediction of multiple variables
- **Attention mechanisms**: (Future development)
- **Physics-informed models**: (Research direction)

### Validation and Uncertainty
- **Cross-validation**: Temporal and spatial validation schemes
- **Uncertainty quantification**: Ensemble predictions (planned)
- **Extreme event validation**: Focus on rare events
- **Multi-basin validation**: Transferability studies

### Advanced Features (Planned)
- **Real-time data integration**: Live weather feeds
- **Ensemble forecasting**: Multiple model predictions
- **Explainable AI**: Model interpretation tools
- **Web interface**: User-friendly prediction platform

## Contributing

### Adding New Experiments
1. Define experiment in `config.yaml` using inheritance
2. Document hydrological rationale and use case
3. Test on validation data before deployment
4. Update documentation and example workflows

### Code Standards
- Follow PEP 8 Python style guidelines
- Include comprehensive docstrings
- Add type hints for function parameters
- Write unit tests for new functionality

### Data Integration
- Support for new watersheds and climate scenarios
- Standardized data formats and naming conventions
- Automated quality control and validation
- Documentation of data sources and processing steps

## Support and Documentation

### Getting Help
- Review example scripts and configurations
- Check documentation in docstrings and comments  
- Validate data processing steps using inspection tools
- Start with simple experiments before advanced configurations

### Troubleshooting
- **Memory issues**: Reduce batch size or window size
- **Training instability**: Adjust learning rate or add regularization
- **Poor performance**: Check data quality and feature selection
- **GPU problems**: Verify CUDA installation and compatibility

### Citation
If you use this code for research, please cite:
```
[Citation information to be added]
```
