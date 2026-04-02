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


Please implmente these functions from train.py to utils.py
from train import (
    EarlyStopping,
    calculate_metrics,
    get_scheduler,
    plot_training_history,
    save_model,
    seed_everything,
)

Then make train_global.py use the functions from utils.py




Traceback (most recent call last):
  File "/projects/standard/kumarv/xu000114/floods_droughts/train_global.py", line 395, in <module>
    main()
  File "/projects/standard/kumarv/xu000114/floods_droughts/train_global.py", line 200, in main
    data_loader = GlobalFloodDroughtDataLoader(
  File "/projects/standard/kumarv/xu000114/floods_droughts/dataloader_global.py", line 107, in __init__
    self.dataset_map = self._load_dynamic_datasets()
  File "/projects/standard/kumarv/xu000114/floods_droughts/dataloader_global.py", line 245, in _load_dynamic_datasets
    raise ValueError(
ValueError: Non-finite values found in data_processed/BlueEarth_hist_scaled_combined_daily.csv: {'streamflow': 8, 'PET': 8, 'ET': 8, 'SUPY': 8, 'WYIE': 8, 'SNOW': 8, 'TWS': 8, 'LZS': 8, 'AGW': 8, 'T2': 8, 'DEWPT': 8, 'PRECIP': 8, 'SWDNB': 8, 'WSPD10': 8, 'LH': 8, 'Area_ac_total': 8}
(imerg_era5) /projects/standard/kumarv/xu000114/floods_droughts$ 

For this bug, please adjust the combined_eddev_flow_daily.py script to handle non-finite values.
First, you need to show when the non-finite values are generated, from exampe the date. 

Then, you can just de


Please also another option to remove the non-finite values by simply removing the dates with non-finite values. And make this option as the default option.
Since  hourly combined file likely has no records for those dates, then just don't have those date in the dataset


python analysis_global.py --config config_global.yaml --experiment daily_global_streamflow_analysis
Why under daily_global_streamflow/analysis_results/test/plot folder, there are not plots for streamflow prediction? Please debug.


# Mar 14, 13:35, 2026
KettleRiverModels'S ET, PET

python -m pdb combined_eddev_flow_daily.py --basin KettleRiverModels --scenario hist_scaled 
In this script the input data already has all zeros for the ET and PET columns.
Check the 
python combine_eddev_flow.py --basin KettleRiverModels --scenario hist_scaled --resolution daily

in the process_flow_data.py
Name: PET, Length: 271752, dtype: float64
(Pdb) np.sum(processed_df['PET'])
np.float64(0.0)
(Pdb) np.sum(processed_df['ET'])
np.float64(0.0)

KettleRiverModels's daily data is actually in hourly resolution. Check the data in original data folder.

Hi Harsh,

Greetings!
I noticed that the following KettleRiverModels datasets appear to be in hourly resolution at the moment:

KettleRiverModels_outlet_RCP8.5_Daily_outputs
KettleRiverModels_outlet_RCP4.5_Daily_outputs
KettleRiverModels_outlet_hist_scaled_Daily_outputs

These datasets should be in daily resolution, and the current hourly format is causing some of the existing scripts to return errors. Would it be possible for you to check and correct them back to the appropriate daily resolution?

Thanks


Next steps: 
After I receive the corrected daily data for KettleRiverModels, I will rerun the 
python process_flow_data.py
combine_eddev_flow.py for KettleRiverModels
python combined_eddev_flow_daily.py --basin $basin --scenario hist_scaled & for KettleRiverModels
All daily_global_* experiments, and streamflow_hmtl_global experiment.

