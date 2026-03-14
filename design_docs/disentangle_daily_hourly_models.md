Train daily model for daily imtermediate variables.
use the predicted daily intermediate variables to train hourly model for streamflow prediction.
All are single task global models.

What is the name of the branch for the new solution? hierarchical_daily_hourly_global

# Train daily model for daily imtermediate variables.
python process_eddev_data.py --all --basin "SnakeSE_Watersheds" --scenario "RCP4.5" &
python process_flow_data.py
python combine_eddev_flow.py --basin $basin --scenario "hist_scaled" --resolution "daily" &


# Create a new script combined_eddev_flow_daily.py 
In this script, we will read the processed data from combine_eddev_flow.py, then convert them to daily data.
load data that generated from combine_eddev_flow.py, for example processed_dir / f"{args.basin}_{args.scenario}_combined{res_tag}.csv"
Convert the hourly data to daily data, using average by day.
Save the daily data to processed_dir / f"{args.basin}_{args.scenario}_combined_daily.csv"

Debug (Done)

Now, I can start to train the daily model for the daily variables. Can I reuse these scripts?
config_global.yaml
train_global.py
inference_global.py
analysis_global.py

For new experiments, we can set the window size to 182 days, and the overlap to 91 days.

daily_intermediate_global:
  <<: *BASE_GLOBAL_CONFIG
  description: >
    Daily model for intermediate variables.
  csv_pattern: "{watershed}_{scenario}_combined_daily.csv"
  window_size: 182
  stride: 91
  target_cols: ["PET", "ET", "SUPY", "WYIE", "SNOW", "TWS", "LZS", "AGW"]
  feature_cols: ["T2", "DEWPT", "PRECIP", "SWDNB", "WSPD10", "LH"]  # adjust if your daily file differs



