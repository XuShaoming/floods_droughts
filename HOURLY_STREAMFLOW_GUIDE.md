# Hourly Streamflow Time Series Configuration Guide

This guide explains how to set window_size and stride parameters for hourly streamflow modeling based on different hydrological objectives.

## Understanding Window Size and Stride

### Window Size (Look-back period)
- **Definition**: Number of previous time steps the model uses to make predictions
- **For hourly data**: Represents hours of historical data
- **Trade-offs**: 
  - Larger = captures longer patterns but requires more memory/computation
  - Smaller = faster training but may miss important long-term patterns

### Stride (Sampling frequency)  
- **Definition**: How many time steps to skip between training samples
- **stride=1**: Use every hour (maximum data, potential overfitting)
- **stride=6**: Use every 6 hours (reduces redundancy, faster training)
- **stride=24**: Use daily samples (focus on daily patterns)

## Recommended Configurations by Use Case

### 1. Short-term Flood Prediction (1-12 hours ahead)
```yaml
hourly_short_term:
  data:
    window_size: 72    # 3 days (72 hours)
    stride: 1          # Every hour
    target_cols: ["streamflow"]
    feature_cols: ["T2", "DEWPT", "PRECIP", "SWDNB", "WSPD10", "LH"]
```
**Rationale**: 
- 72 hours captures storm event development and peak response
- stride=1 preserves all temporal resolution for critical flood timing
- Includes precipitation and temperature for runoff modeling

### 2. Daily Cycle Modeling (snowmelt, ET patterns)
```yaml
hourly_daily_cycles:
  data:
    window_size: 168   # 1 week (7 days × 24 hours)
    stride: 6          # Every 6 hours  
    target_cols: ["streamflow"]
    feature_cols: null  # Use all features
```
**Rationale**:
- 168 hours captures weekly patterns and multiple daily cycles
- stride=6 reduces data redundancy while preserving diurnal patterns
- Full feature set captures complex daily interactions

### 3. Flood Event Analysis (multi-day events)
```yaml
hourly_flood_events:
  data:
    window_size: 120   # 5 days (120 hours)
    stride: 12         # Every 12 hours
    target_cols: ["streamflow"] 
    feature_cols: ["PRECIP", "T2", "SWDNB", "WSPD10"]
```
**Rationale**:
- 120 hours captures full flood event lifecycle (rise, peak, recession)
- stride=12 focuses on twice-daily patterns, reduces noise
- Selected features most relevant to flood generation

### 4. Baseflow and Seasonal Trends
```yaml
hourly_baseflow:
  data:
    window_size: 360   # 15 days (360 hours)
    stride: 24         # Daily sampling
    target_cols: ["streamflow"]
    feature_cols: ["T2", "DEWPT", "ET", "SWDNB"]
```
**Rationale**:
- 360 hours captures seasonal baseflow trends
- stride=24 focuses on daily changes, ignores hourly noise
- Features related to evapotranspiration and energy balance

## Window Size Guidelines by Watershed Type

### Urban Watersheds (Fast response)
- **Short events**: 24-48 hours (window_size: 24-48)
- **Multiple storms**: 72-120 hours (window_size: 72-120)
- **Stride**: 1-3 hours (preserve rapid response timing)

### Rural/Natural Watersheds (Slower response)
- **Storm events**: 72-168 hours (window_size: 72-168)
- **Seasonal patterns**: 168-720 hours (window_size: 168-720)
- **Stride**: 6-12 hours (daily patterns more important)

### Large River Basins (Very slow response)
- **Flood waves**: 240-720 hours (window_size: 240-720)
- **Seasonal flows**: 720-2160 hours (window_size: 720-2160)
- **Stride**: 12-24 hours (focus on daily/weekly patterns)

## Memory and Performance Considerations

### Computational Requirements
```
Memory Usage ∝ window_size × batch_size × num_features
Training Time ∝ (total_samples / stride) × window_size
```

### Recommended Combinations
| Window Size | Stride | Batch Size | Use Case |
|-------------|--------|------------|----------|
| 24-72       | 1-3    | 32-64      | Short-term prediction |
| 72-168      | 6-12   | 16-32      | Daily patterns |
| 168-360     | 12-24  | 8-16       | Weekly/seasonal |
| 360+        | 24+    | 4-8        | Long-term trends |

## Feature Selection by Time Scale

### Hourly Features (stride=1-3)
- **Essential**: PRECIP, T2, streamflow
- **Important**: DEWPT, WSPD10, SWDNB
- **Secondary**: LH, pressure variables

### Daily Features (stride=6-24)  
- **Essential**: PRECIP (accumulated), T2 (mean/max/min)
- **Important**: ET, SWDNB (daily totals)
- **Secondary**: DEWPT, WSPD10

## Example Usage

```bash
# Quick flood prediction (3-day memory, hourly)
python train.py --experiment hourly_short_term

# Daily cycle modeling (1-week memory, 6-hour stride)  
python train.py --experiment hourly_daily_cycles

# Flood event analysis (5-day memory, 12-hour stride)
python train.py --experiment hourly_flood_events

# Baseflow modeling (15-day memory, daily stride)
python train.py --experiment hourly_baseflow

# Custom configuration
python train.py --experiment hourly_short_term --override data.window_size=96 data.stride=2
```

## Validation Strategy

### Time Series Cross-Validation
1. **Sequential splits**: Respect temporal order
2. **Seasonal validation**: Test on different seasons  
3. **Event-based validation**: Test on extreme events separately

### Performance Metrics by Time Scale
- **Hourly**: Nash-Sutcliffe Efficiency (NSE), RMSE
- **Daily**: Peak flow timing, volume errors
- **Seasonal**: Bias in seasonal totals, baseflow index

## Best Practices

1. **Start with domain knowledge**: Consider watershed response time
2. **Experiment incrementally**: Try window_size in powers of 2 (24, 48, 96, 192...)
3. **Balance stride with patterns**: Don't stride over important cycles
4. **Monitor memory usage**: Adjust batch_size with window_size
5. **Validate on events**: Test performance on actual flood/drought events

This configuration approach ensures your LSTM model captures the right temporal patterns for accurate streamflow prediction!
