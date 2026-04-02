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


# Apr 2, Meeting notes:

Steps:
- We want to train the daily IMV models first and get the predicted daily IMV outputs for the training, validation, and testing splits. (Done)
- We need to combine the predicted daily IMV outputs from three splits together into one for each watershed. (Done) (Daily scale)
- We need to interpolate the daily IMV outputs to hourly resolution, and combine them with the hourly streamflow data and hourly eddv data. (Done) (Hourly scale) for each of waterheds. 
-- training ,validation, and testing (three splits)
- We need to combien three splits together to one file for each watershed, and add a column to indicate the split name. (Done)


  dataset_splits:
    train:
      - watersheds: ["BlueEarth", "LeSueur", "Watonwan", "KettleRiverModels", "LittleFork", "Zumbro", "SnakeSE"]
        scenarios: ["hist_scaled"]
        start: "1980-01-01 00:00:00"
        end: "2000-12-31 23:00:00"
    val:
      - watersheds: ["BlueEarth", "LeSueur", "Watonwan", "KettleRiverModels", "LittleFork", "Zumbro", "SnakeSE"]
        scenarios: ["hist_scaled"]
        start: "1975-01-01 00:00:00"
        end: "1979-12-31 23:00:00"
    test:
      - watersheds: ["BlueEarth", "LeSueur", "Watonwan", "KettleRiverModels", "LittleFork", "Zumbro", "SnakeSE"]
        scenarios: ["hist_scaled"]
        start: "2001-01-01 00:00:00"
        end: "2005-12-31 23:00:00"


1975-01-01 00:00:00 - 2005-12-31 23:00:00


Next step: (Now), I will probably finihsh before the next meeting.
Training, validation and testing
inputs (Houly observed weather drivers + observed static watershed attributes + predicted daily IMVs intoplrated into hourly resolution) -> Model -> outputs(hourly streamflow predictions)
--> how good is the perfomance.
---> Peformance is good? How can improve the performance furhter?
---> Peformance is notgood? How can improve the performance furhter?

How can improve the performance furhter?
Answer, improve the model perforance on IMVs, because of previous results.
-> Conditional minibatch learning or some other algorihtms to improve the IMV predictions.

Previously, 
Training, validation and testing
inputs (Houly observed weather drivers + observed static watershed attributes + groudtruth daily IMVs intoplrated into hourly resolution) -> Model -> outputs(hourly streamflow predictions)
--> prefect results.


Now: datasets have been prepared.


# Apr 2, 12:20, 2026

## Build the hourly streamflow using the daily global model IMV outputs as additional inputs.
- First, I need to list all the IMV training, validation, and testing outputs. (Done)
- Second, I need to process the IMV outputs to combine them into a single file for each watershed and split. (Done)
-- combine_imv_outputs.py. (Done)
-- Check the inference and analysis scripts to see where the IMV outputs are saved.
--- Mainly check the format of the file name so I can reuse the code to retrieve the file paths.
--- In inference_global.py, 

            recon_filename = f"{watershed_key}_{split_name}_reconstructed_{method}.{file_format}"
            recon_path = os.path.join(output_dir, recon_filename)
            save_dataframe(recon_df, recon_path, file_format)

so, now I need to implment a script to combine all the IMV outputs for each specific watershed.
I have watersheds: ["BlueEarth", "LeSueur", "Watonwan", "KettleRiverModels", "LittleFork", "Zumbro", "SnakeSE"] and I have training, validation, and testing splits. So, this step will results total 7 (watersheds) * 3 (splits) = 21 files for the IMV outputs. Each file will contains all the IMVs including ["T2", "DEWPT", "PRECIP", "SWDNB", "WSPD10", "LH"] and ["streamflow"] in daily resolution. 

Where to save the combined IMV outputs? data_processed subfolder under experiments. The path can be experiments/data_processed/{watershed}_{split_name}_imv_outputs.csv

The prompt:
Please implement a script to combine all the IMV outputs for each specific watershed. The script should read the IMV outputs from the inference step, combine them into a single file for each watershed and split, and save the combined files in the experiments/data_processed/ directory with the naming convention {watershed}_{split_name}_imv_outputs.csv.

For example, I have watersheds: ["BlueEarth", "LeSueur", "Watonwan", "KettleRiverModels", "LittleFork", "Zumbro", "SnakeSE"] and I have training, validation, and testing splits. So, this step will results total 7 (watersheds) * 3 (splits) = 21 files for the IMV outputs. Each file will contains all the IMVs including ["T2", "DEWPT", "PRECIP", "SWDNB", "WSPD10", "LH"] and ["streamflow"] in daily resolution. 

Each file can have columns like timestamp,pred_TWS,obs_TWS,pred_LZS,obs_LZS,..., scenario 

You can take reference this code snippet from inference_global.py to write the codes to read the IMV outputs:

            recon_filename = f"{watershed_key}_{split_name}_reconstructed_{method}.{file_format}"
            recon_path = os.path.join(output_dir, recon_filename)
            recon_df = load_dataframe(recon_path, file_format)
The name of the script can be combine_imv_outputs.py.

- Third, I need to check when to intoplate the daily IMV and streamflow into hourly data, so as to be the additional inputs for the hourly streamflow model.
-- Check the previous solution.
--- Previous solutions is implmented in process_flow_data.py
### Implement the combine_daily_imv_outputs_hourly_streamflow_hourly_eddv.py
combine_daily_imv_outputs_hourly_streamflow_hourly_eddv.py
In this script, we will read the combined daily IMV outputs, the hourly streamflow data, and hourly eddv data, then combine them together and save the combined data to a new file. The daily IMV outputs will be interpolated to hourly resolution, and then merged with the hourly streamflow and eddv data based on timestamp. You can take reference from the previous solution in process_flow_data.py for the interpolation part and combine_eddev_flow.py for the merging  part on the eddv data.

This will results 21 files for each watershed and split, with hourly data for both IMVs and streamflow, and eddv data. Each file will be saved in the experiments/data_processed/ directory with the naming convention {watershed}_{split_name}_combined_hourly.csv.

Please read a few lines of this file to check if the data is correct. For example, you can read SnakeSE_test_combined_hourly.csv. I think the PET and obs_PET columns should be the same. This would be same for other intermediate variables including  ["T2", "DEWPT", "PRECIP", "SWDNB", "WSPD10", "LH"]. Please check if they match. If they don't match, Please explan why they don't match.

As obs_* are essentially the same as the orignal data, combine_daily_imv_outputs_hourly_streamflow_hourly_eddv.py does not need to save the obs_* columns in the combined file, keep others the same. So the final columns can be Datetime,scenario,streamflow,PET,ET,SUPY,WYIE,SNOW,TWS,LZS,AGW,T2,DEWPT,PRECIP,SWDNB,WSPD10,LH,Area_ac_total,pred_AGW,pred_ET,pred_LZS,pred_PET,pred_SNOW,pred_SUPY,pred_TWS,pred_WYIE,pred_streamflow,


- Third, I need to create a training, inference, and analysis script for this experiments.
-- For training, let see if I can resuse the original training script.
I should reuse the dataloader in the train_global.py, this means I need to combine the train, validation, and testing data together to one file. The file name can be {watershed}_{scenario}_combined.csv. 
So for this purpose, please make the combine_daily_imv_outputs_hourly_streamflow_hourly_eddv.py to also combine the train, validation, and testing data together to one file for each watershed. The file name can be {watershed}_{scenario}_combined.csv. This will results 7 files for each watershed, with hourly data for both IMVs and streamflow, and eddv data. Each file will be saved in the experiments/data_processed/ directory with the naming convention {watershed}_{scenario}_combined.csv. In this file, also add a column for split_name to indicate whether the record is from training, validation, or testing split. Please make sure the data is sorted by timestamp after combining the three splits together.

I need to compare the format of the combined csv files.

Datetime,streamflow,PET,ET,SUPY,WYIE,SNOW,TWS,LZS,AGW,T2,DEWPT,PRECIP,SWDNB,WSPD10,LH,Area_ac_total

Datetime,scenario,streamflow,PET,ET,SUPY,WYIE,SNOW,TWS,LZS,AGW,T2,DEWPT,PRECIP,SWDNB,WSPD10,LH,Area_ac_total,pred_AGW,pred_ET,pred_LZS,pred_PET,pred_SNOW,pred_SUPY,pred_TWS,pred_WYIE,pred_streamflow,split_name

- 

Now, I need to prepare the experiments in the config_global.yaml for training, inference, and analysis.

Please check if hourly_global_streamflow is set correctly in the config_global.yaml so that I can reuse the train_global.py, inference_global.py, and analysis_global.py for this experiment. If not, please adjust the config_global.yaml to make sure the hourly_global_streamflow is set correctly. If yes, please also add hourly_global_streamflow_inference and hourly_global_streamflow_analysis in the config_global.yaml for the inference and analysis steps.

Experiment:
hourly_global_streamflow



