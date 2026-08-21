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
- **Comprehensive metrics**: MSE, RMSE, MAE, R², MAPE, NSE, KGE and its
  correlation/variability/bias components, sigma-normalized RMSE, and signed bias
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

### Global Multi-Watershed Modeling
- **`config_global.yaml`**: Defines multi-watershed experiments (e.g., `streamflow_global_exp1`) with explicit `scenarios` (`hist_scaled`, `RCP4.5`, `RCP8.5`), scenario date ranges, and `dataset_splits` that pair watershed lists with scenario/date filters for train/val/test.
- **`dataloader_global.py`**: Loads all requested `{watershed}_{scenario}_combined.csv` files, filters them according to the split definitions, and injects static watershed attributes from `hspf_CAT_attributes.csv`.
- **`train_global.py`**: Trains a single LSTM across arbitrary watershed/scenario combinations, embeds static attributes, logs TensorBoard metrics, and saves learning curves to each experiment folder.
- **`inference_global.py`**: Recreates the global data pipeline for evaluation, supports optional `--watersheds` / `--scenarios` filtering, and writes per-split metrics plus raw predictions/targets.

```bash
# Train the global model with static attribute embeddings
python train_global.py --config config_global.yaml --experiment streamflow_global_exp1
# Evaluate the trained global checkpoint on the test split
python inference_global.py --model-dir experiments/global_models/streamflow_global_exp1 --dataset test --scenarios RCP4.5
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

#### Paired extreme-flow model comparison

`compare_flow_extremes.py` uses the same H-LSTM, MR-STF, and MR-PTF sources
configured in `plot_reconstruction_comparison.py`. It aligns all models to
common timestamps, uses one observed series and shared basin Q10/Q90/Q95
thresholds, and reports paired regime errors, bootstrap intervals, flood-event
matching, flow-duration curves, representative flood hydrographs, and a 3 × 4
hydrograph figure containing the largest normalized event from every basin.

```bash
conda run -n imerg_era5 python compare_flow_extremes.py
```

Create publication-ready versions of the main comparison, basin-level
performance, watershed-level robustness, and representative flood-event
figures (PNG, PDF, and SVG):

```bash
conda run -n imerg_era5 python plot_paper_flow_extremes.py
```

#### Daily IMV and streamflow evaluation

`analyze_daily_global_results.py` evaluates every `daily_global_*` model that
uses `T2`, `DEWPT`, `PRECIP`, `SWDNB`, `WSPD10`, and `LH`. It calculates
watershed-level diagnostics for train, validation, and test reconstructions;
summarizes KGE as an unweighted watershed mean and sample standard deviation;
and creates 4 × 3 test-period time-series figures for all eight IMVs and
streamflow. It also writes a KGE comparison figure, a test-basin heatmap, data
audit tables, and a documented Markdown report with suggested paper captions.

```bash
# Full paper-facing analysis in PNG and PDF
conda run --no-capture-output -n imerg_era5 python analyze_daily_global_results.py

# Faster PNG-only export
conda run --no-capture-output -n imerg_era5 \
  python analyze_daily_global_results.py --formats png

# Show every available option
conda run --no-capture-output -n imerg_era5 \
  python analyze_daily_global_results.py --help
```

Outputs are written by default to
`experiments/daily_global_comparison/analysis_results/`. The main paper table
is `tables/daily_global_kge_summary.csv`; the exact basin metrics are retained
in `tables/daily_global_metrics_by_basin.csv`.

#### Hourly versus daily streamflow comparison

`compare_hourly_daily_streamflow.py` compares the hourly H-LSTM experiment
`hourly_global_streamflow_no_IMVs`, the hourly MR-STF experiment
`hourly_global_streamflow_pred_streamflow_day_shift_1`, and the daily D-LSTM
experiment `daily_global_streamflow` during their common test period. It uses the
project's existing calendar-day repetition method to place each daily
prediction on the hourly grid, evaluates all three models against the same hourly
reference, and repeats the comparison after daily averaging. The analysis
reports RMSE, NSE, KGE, high-flow error, bias, and within-day variation skill;
it creates full-period 4 x 3 hydrographs, peak-event zooms, and paired
watershed comparison figures.

```bash
# Full analysis and PNG/PDF figures
conda run --no-capture-output -n imerg_era5 \
  python compare_hourly_daily_streamflow.py

# Faster PNG-only export
conda run --no-capture-output -n imerg_era5 \
  python compare_hourly_daily_streamflow.py --formats png

# Show paths, watershed selection, formats, and other options
conda run --no-capture-output -n imerg_era5 \
  python compare_hourly_daily_streamflow.py --help
```

Outputs are written by default to
`experiments/hourly_daily_streamflow_comparison/test_results/`. See
`comparison_notes.md` there for the methods, exact aggregate results,
interpretation cautions, and suggested paper captions.

#### Four-model paper tables and flood-event figure

`compare_four_model_test_performance.py` adds the daily D-LSTM to the
leakage-controlled H-LSTM, MR-STF, and MR-PTF paper comparisons. It recalculates
all four models on their exact common hourly test window, writes overall,
extreme-flow, flood-event, and KGE-component tables to one documented Markdown
file, saves the unrounded basin values as CSV files, and creates a 4 x 3
representative-flood-event figure with all models and watersheds. It also writes
12 full-test-period figures under `reconstructions/`, with model curves and table
columns ordered D-LSTM, H-LSTM, MR-STF, and MR-PTF while retaining each model's
established color.

```bash
conda run --no-capture-output -n imerg_era5 \
  python compare_four_model_test_performance.py
```

Outputs are written to
`experiments/hourly_daily_four_model_comparison/test_results/` by default.

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
