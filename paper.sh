#!/usr/bin/env bash

# Valid --device choices: auto, cpu, 0, 1, ... or cuda:N
# Numeric choices are logical CUDA indices visible to this process.

# python process_flow_data.py

# basin_names ={
#     'WatonwanR_Watersheds': 'Watonwan',
#     'LeSueurR_Watersheds': 'LeSueur',
#     'KettleR_Watersheds': 'KettleRiverModels',
#     'BlueEarthR_Watersheds': 'BlueEarth',
#     'LittleFork_Watersheds': 'LittleFork',
#     'SnakeSE_Watersheds': 'SnakeSE',
#     'Zumbro_Watersheds': 'Zumbro',
#     'Cloquet_Watersheds': 'Cloquet',
#     'Sauk_Watersheds': 'Sauk',
#     'SFCrow_Watersheds': 'SFCrow',
#     'WildRice_Watersheds': 'WildRiceMarsh',
#     'TwoRivers_Watersheds': 'TwoRivers'
# }

# python process_eddev_data.py --all --basin "Cloquet_Watersheds" --scenario "Historical" &
# python process_eddev_data.py --all --basin "Cloquet_Watersheds" --scenario "RCP4.5" &
# python process_eddev_data.py --all --basin "Cloquet_Watersheds" --scenario "RCP8.5" &
# python process_eddev_data.py --all --basin "Sauk_Watersheds" --scenario "Historical" &
# wait
# python process_eddev_data.py --all --basin "Sauk_Watersheds" --scenario "RCP4.5" &
# python process_eddev_data.py --all --basin "Sauk_Watersheds" --scenario "RCP8.5" &
# python process_eddev_data.py --all --basin "SFCrow_Watersheds" --scenario "Historical" &
# python process_eddev_data.py --all --basin "SFCrow_Watersheds" --scenario "RCP4.5" &
# wait
# python process_eddev_data.py --all --basin "WildRiceMarsh_Watersheds" --scenario "Historical" &
# python process_eddev_data.py --all --basin "WildRiceMarsh_Watersheds" --scenario "RCP4.5" &
# python process_eddev_data.py --all --basin "WildRiceMarsh_Watersheds" --scenario "RCP8.5" &


# python process_eddev_data.py --all --basin "TwoRivers_Watersheds" --scenario "RCP4.5" &
# python process_eddev_data.py --all --basin "TwoRivers_Watersheds" --scenario "RCP8.5" &

# wait
# python combine_eddev_flow.py --basin "Cloquet" --scenario "hist_scaled" &
# python combine_eddev_flow.py --basin "Cloquet" --scenario "RCP4.5" & 
# python combine_eddev_flow.py --basin "Cloquet" --scenario "RCP8.5" &
# python combine_eddev_flow.py --basin "Sauk" --scenario "hist_scaled" &
# wait
# python combine_eddev_flow.py --basin "Sauk" --scenario "RCP4.5" &
# python combine_eddev_flow.py --basin "Sauk" --scenario "RCP8.5" &
# python combine_eddev_flow.py --basin "SFCrow" --scenario "hist_scaled" &
# python combine_eddev_flow.py --basin "SFCrow" --scenario "RCP4.5" &
# wait
# python combine_eddev_flow.py --basin "WildRiceMarsh" --scenario "hist_scaled" &
# python combine_eddev_flow.py --basin "WildRiceMarsh" --scenario "RCP4.5" &
# python combine_eddev_flow.py --basin "WildRiceMarsh" --scenario "RCP8.5" &


# for basin in Cloquet Sauk SFCrow TwoRivers WildRiceMarsh; do
#     python combined_eddev_flow_daily.py --basin $basin --scenario hist_scaled &
#     python combined_eddev_flow_daily.py --basin $basin --scenario RCP4.5 &
#     python combined_eddev_flow_daily.py --basin $basin --scenario RCP8.5 &
#     wait
# done

# python process_eddev_data.py --all --basin "TwoRivers_Watersheds" --scenario "Historical"
# python combine_eddev_flow.py --basin "TwoRivers" --scenario "hist_scaled"
# python combine_eddev_flow.py --basin "TwoRivers" --scenario "RCP4.5"
# python combine_eddev_flow.py --basin "TwoRivers" --scenario "RCP8.5"
# python combined_eddev_flow_daily.py --basin TwoRivers --scenario hist_scaled
# python combined_eddev_flow_daily.py --basin TwoRivers --scenario RCP4.5
# python combined_eddev_flow_daily.py --basin TwoRivers --scenario RCP8.5


# wait
# python train_global.py --config config_global.yaml --experiment daily_global_streamflow --device 0 &
# python train_global.py --config config_global.yaml --experiment daily_global_PET --device 1 &
# python train_global.py --config config_global.yaml --experiment daily_global_ET --device 2 &
# python train_global.py --config config_global.yaml --experiment daily_global_SUPY --device 3 &
# wait
# python train_global.py --config config_global.yaml --experiment daily_global_WYIE --device 0 &
# python train_global.py --config config_global.yaml --experiment daily_global_SNOW --device 1 &
# python train_global.py --config config_global.yaml --experiment daily_global_TWS --device 2 &
# python train_global.py --config config_global.yaml --experiment daily_global_LZS --device 3 &
# python train_global.py --config config_global.yaml --experiment daily_global_AGW --device 1 &
# wait
# Re-run inference and analysis to regenerate metrics; training remains disabled.
# Valid --dataset choices: train, val, test, all
python inference_global.py --config config_global.yaml --experiment daily_global_streamflow_inference --dataset all --device 0 &
python inference_global.py --config config_global.yaml --experiment daily_global_PET_inference --dataset all --device 1 &
python inference_global.py --config config_global.yaml --experiment daily_global_ET_inference --dataset all --device 2 &
python inference_global.py --config config_global.yaml --experiment daily_global_SUPY_inference --dataset all --device 3 &
wait
python inference_global.py --config config_global.yaml --experiment daily_global_WYIE_inference --dataset all --device 0 &
python inference_global.py --config config_global.yaml --experiment daily_global_SNOW_inference --dataset all --device 1 &
python inference_global.py --config config_global.yaml --experiment daily_global_TWS_inference --dataset all --device 2 &
python inference_global.py --config config_global.yaml --experiment daily_global_LZS_inference --dataset all --device 3 &
python inference_global.py --config config_global.yaml --experiment daily_global_AGW_inference --dataset all --device 2 &
wait
# Valid --split choices: train, val, test, all
python analysis_global.py --config config_global.yaml --experiment daily_global_streamflow_analysis --split all &
python analysis_global.py --config config_global.yaml --experiment daily_global_PET_analysis --split all &
python analysis_global.py --config config_global.yaml --experiment daily_global_ET_analysis --split all &
python analysis_global.py --config config_global.yaml --experiment daily_global_SUPY_analysis --split all &
python analysis_global.py --config config_global.yaml --experiment daily_global_WYIE_analysis --split all &
wait
python analysis_global.py --config config_global.yaml --experiment daily_global_SNOW_analysis --split all &
python analysis_global.py --config config_global.yaml --experiment daily_global_TWS_analysis --split all &
python analysis_global.py --config config_global.yaml --experiment daily_global_LZS_analysis --split all &
python analysis_global.py --config config_global.yaml --experiment daily_global_AGW_analysis --split all &
wait



# # Hourly global streamflow with daily IMV inputs
# python combine_daily_imv_outputs.py
# python combine_daily_imv_outputs_hourly_streamflow_hourly_eddy.py

# wait
# python train_global.py --config config_global.yaml --experiment hourly_global_streamflow_observed_IMVs --device 0 &
# python train_global.py --config config_global.yaml --experiment hourly_global_streamflow_no_IMVs --device 1 &
# python train_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_PET --device 2 &
# python train_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_ET --device 3 &
# wait
# python train_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_SUPY --device 0 &
# python train_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_WYIE --device 1 &
# python train_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_SNOW --device 2 &
# python train_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_TWS --device 3 &
# wait
# python train_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_LZS --device 0 &
# python train_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_AGW --device 1 &
# python train_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_streamflow --device 2 &
# wait

# Valid --dataset choices: train, val, test, all
python inference_global.py --config config_global.yaml --experiment hourly_global_streamflow_observed_IMVs_inference --dataset test --device 0 &
python inference_global.py --config config_global.yaml --experiment hourly_global_streamflow_no_IMVs_inference --dataset test --device 1 &
python inference_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_PET_inference --dataset test --device 2 &
python inference_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_ET_inference --dataset test --device 3 &
wait
python inference_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_SUPY_inference --dataset test --device 0 &
python inference_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_WYIE_inference --dataset test --device 1 &
python inference_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_SNOW_inference --dataset test --device 2 &
python inference_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_TWS_inference --dataset test --device 3 &
wait
python inference_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_LZS_inference --dataset test --device 0 &
python inference_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_AGW_inference --dataset test --device 1 &
python inference_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_streamflow_inference --dataset test --device 2 &
wait

# Valid --split choices: train, val, test, all
python analysis_global.py --config config_global.yaml --experiment hourly_global_streamflow_observed_IMVs_analysis --split test
python analysis_global.py --config config_global.yaml --experiment hourly_global_streamflow_no_IMVs_analysis --split test &
python analysis_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_PET_analysis --split test &
python analysis_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_ET_analysis --split test &
python analysis_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_SUPY_analysis --split test &
python analysis_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_WYIE_analysis --split test &
wait
python analysis_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_SNOW_analysis --split test &
python analysis_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_TWS_analysis --split test &
python analysis_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_LZS_analysis --split test &
python analysis_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_AGW_analysis --split test &
python analysis_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_streamflow_analysis --split test &
wait
