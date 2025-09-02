# Flood/Drought Prediction Workflow

```mermaid
flowchart TD
    %% Data Sources
    A1[EDDEV1 Climate Data<br/>36km resolution<br/>T2, DEWPT, PRECIP, etc.]
    A2[HSPF Streamflow Data<br/>Hourly simulations]
    A3[Watershed Boundaries<br/>HUC8/HUC12]
    
    %% Data Processing
    B1[Spatial Processing<br/>KNN + IDW Downscaling<br/>→ HUC12 → HUC8]
    B2[Temporal Alignment<br/>Hourly Time Series<br/>Weather + Streamflow]
    
    %% Combined Dataset
    C1[Combined Dataset<br/>Weather & Streamflow<br/>Hourly Time Series]
    
    %% Configuration & Data Prep
    D1[Configuration<br/>config.yaml<br/>Experiment Setup]
    D2[Data Loading<br/>Time Windows<br/>Normalization<br/>Train/Val/Test Split]
    
    %% Model Training
    E1[LSTM<br/>Training Pipeline<br/>Loss Optimization]
    E2[Model Checkpointing<br/>Early Stopping<br/>Best Model Selection]
    
    %% Model Evaluation
    F1[Model Inference<br/>Prediction Generation]
    F2[Performance Metrics<br/>RMSE, NSE, KGE<br/>Flood/Drought Analysis]
    
    %% Final Results
    G1[Results<br/>Predictions<br/>Visualizations<br/>Performance Reports]
    
    %% Flow connections
    A1 --> B1
    A2 --> B1
    A3 --> B1
    B1 --> B2
    B2 --> C1
    
    C1 --> D2
    D1 --> D2
    D2 --> E1
    E1 --> E2
    
    E2 --> F1
    F1 --> F2
    F2 --> G1
    
    %% Styling
    classDef dataSource fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef processing fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef config fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef training fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef evaluation fill:#fff8e1,stroke:#fbc02d,stroke-width:2px
    classDef results fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    
    class A1,A2,A3 dataSource
    class B1,B2 processing
    class C1 processing
    class D1 config
    class D2 config
    class E1,E2 training
    class F1,F2 evaluation
    class G1 results
```

<!-- ## Key Workflow Steps

### 1. **Data Integration**
- **Climate Data**: EDDEV1 meteorological variables at 36km resolution
- **Streamflow Data**: HSPF hourly simulation outputs
- **Spatial Processing**: KNN grid selection + IDW downscaling to watershed level

### 2. **Data Preparation**
- **Temporal Alignment**: Combine weather and streamflow into hourly time series
- **Windowing**: Create sequences for LSTM input
- **Normalization**: Standardize features for model training

### 3. **Model Development**
- **LSTM**: Sequential modeling for streamflow prediction
- **Multiple Quantiles**: Capture uncertainty and extreme events (floods/droughts)
- **Training Pipeline**: Optimization with early stopping and checkpointing

### 4. **Evaluation & Results**
- **Multi-quantile Predictions**: Generate uncertainty bounds
- **Performance Metrics**: RMSE, NSE, KGE for model validation
- **Extreme Event Analysis**: Flood and drought characterization

## Key Features
- **Spatial Downscaling**: From 36km climate grids to watershed-specific predictions
- **Uncertainty Quantification**: Multiple quantiles for risk assessment
- **Extreme Event Focus**: Specialized modeling for floods and droughts
- **Scalable Pipeline**: Configurable for multiple watersheds and scenarios -->
