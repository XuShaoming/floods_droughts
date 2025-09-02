# Data Preprocessing Pipeline for Combined Dataset Generation

```mermaid
flowchart LR
    %% Input Data Sources
    A1[EDDEV1 Climate Data<br/>36km x 36km grid<br/>T2, DEWPT, PRECIP<br/>SWDNB, WSPD10, LH]
    A2[HSPF Streamflow Data<br/>Hourly simulations<br/>Multiple scenarios]
    A3[Watershed Shapefiles<br/>HUC8/HUC12 boundaries<br/>Spatial polygons]
    
    %% Processing Scripts
    B1[process_eddev_data.py<br/>Climate Processing]
    B2[process_flow_data.py<br/>Streamflow Processing]
    
    %% Detailed Processing Steps for Climate Data
    C1[KNN Grid Selection<br/>Select relevant points]
    C2[IDW Downscaling<br/>36km to HUC12]
    C3[Area Aggregation<br/>HUC12 to HUC8]
    
    %% Detailed Processing Steps for Streamflow Data
    D1[Data Integration<br/>Merge hourly flows]
    D2[Quality Control<br/>Validation & cleaning]
    D3[Temporal Interpolation<br/>Fill missing hours]
    
    %% Final Integration
    E1[combine_eddev_flow.py<br/>Data Integration]
    E2[Temporal Alignment<br/>Quality validation]
    
    %% Final Output
    F1[Combined Dataset<br/>CSV File<br/>Weather + Streamflow<br/>Hourly Time Series]
    
    %% Flow Connections - Climate Path
    A1 --> B1
    A3 --> B1
    B1 --> C1
    C1 --> C2
    C2 --> C3
    
    %% Flow Connections - Streamflow Path
    A2 --> B2
    B2 --> D1
    D1 --> D2
    D2 --> D3
    
    %% Flow Connections - Integration
    C3 --> E1
    D3 --> E1
    E1 --> E2
    E2 --> F1
    
    %% Styling
    classDef inputData fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef processing fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef climateStep fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef streamStep fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef integration fill:#fff8e1,stroke:#fbc02d,stroke-width:2px
    classDef output fill:#fce4ec,stroke:#c2185b,stroke-width:3px
    
    class A1,A2,A3 inputData
    class B1,B2 processing
    class C1,C2,C3 climateStep
    class D1,D2,D3 streamStep
    class E1,E2 integration
    class F1 output
```

## Data Preprocessing Algorithm Details

### Climate Data Processing (process_eddev_data.py)

#### Step 1: K-Nearest Neighbors Grid Selection
- **Input**: 36km x 36km EDDEV1 climate grid, watershed polygons
- **Algorithm**: For each watershed, find the k nearest grid points
- **Purpose**: Reduce computational load by focusing on relevant grid cells
- **Output**: Subset of climate grid points per watershed

#### Step 2: Inverse Distance Weighting (IDW) Downscaling
- **Input**: Selected grid points, HUC12 sub-watershed boundaries
- **Algorithm**: 
  ```
  Value_HUC12 = Σ(Value_grid × Weight_distance) / Σ(Weight_distance)
  Weight = 1 / distance²
  ```
- **Purpose**: Downscale from 36km resolution to sub-watershed level
- **Output**: Weather variables interpolated to each HUC12

#### Step 3: Area-weighted Aggregation
- **Input**: HUC12-level weather data, HUC8 boundaries
- **Algorithm**:
  ```
  Value_HUC8 = Σ(Value_HUC12 × Area_HUC12) / Total_Area_HUC8
  ```
- **Purpose**: Aggregate sub-watersheds to main watershed scale
- **Output**: Basin-scale meteorological time series

### Streamflow Data Processing (process_flow_data.py)

#### Step 1: Multi-source Data Integration
- **Input**: HSPF hourly flows, daily metrics, scenario files
- **Algorithm**: Merge by datetime and scenario identifiers
- **Purpose**: Combine different temporal resolutions and data sources
- **Output**: Unified streamflow dataset

#### Step 2: Quality Control & Validation
- **Algorithm**: 
  - Detect missing values and gaps
  - Identify outliers using statistical thresholds
  - Flag inconsistent time stamps
- **Purpose**: Ensure data quality before processing
- **Output**: Quality-flagged streamflow data

#### Step 3: Temporal Interpolation
- **Algorithm**: Linear or spline interpolation for missing hours
- **Purpose**: Create continuous hourly time series
- **Output**: Gap-filled streamflow time series

### Final Data Integration (combine_eddev_flow.py)

#### Temporal Alignment
- **Algorithm**: Inner join on datetime index
- **Validation**: Check for temporal gaps and misalignments
- **Output**: Synchronized weather-streamflow dataset

#### Quality Assurance
- **Coverage Check**: Ensure adequate data coverage for modeling
- **Statistical Summary**: Generate descriptive statistics
- **Error Detection**: Flag potential data issues

## Final Combined Dataset Structure

| Column Type | Variables | Description |
|------------|-----------|-------------|
| **Climate** | T2, DEWPT, PRECIP, SWDNB, WSPD10, LH | Basin-averaged hourly meteorological data |
| **Streamflow** | Flow (various scenarios) | Hourly streamflow simulations |
| **Temporal** | DateTime index | Synchronized time stamps |
| **Metadata** | Scenario, Quality flags | Data provenance and quality indicators |

**Output File**: `KettleRiverModels_hist_scaled_combined.csv`
**Format**: Hourly time series ready for LSTM training
**Coverage**: Complete temporal coverage with quality validation
