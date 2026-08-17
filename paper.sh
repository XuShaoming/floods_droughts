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



# python train_global.py --config config_global.yaml --experiment daily_global_streamflow
# python inference_global.py --config config_global.yaml --experiment daily_global_streamflow_inference
# python analysis_global.py --config config_global.yaml --experiment daily_global_streamflow_analysis

# wait
# python train_global.py --config config_global.yaml --experiment daily_global_PET &
# python train_global.py --config config_global.yaml --experiment daily_global_ET &
# python train_global.py --config config_global.yaml --experiment daily_global_SUPY &
# python train_global.py --config config_global.yaml --experiment daily_global_WYIE &
# wait
# python train_global.py --config config_global.yaml --experiment daily_global_SNOW &
# python train_global.py --config config_global.yaml --experiment daily_global_TWS &
# python train_global.py --config config_global.yaml --experiment daily_global_LZS &
# python train_global.py --config config_global.yaml --experiment daily_global_AGW &
# wait
python inference_global.py --config config_global.yaml --experiment daily_global_PET_inference & 
python inference_global.py --config config_global.yaml --experiment daily_global_ET_inference &
python inference_global.py --config config_global.yaml --experiment daily_global_SUPY_inference &
python inference_global.py --config config_global.yaml --experiment daily_global_WYIE_inference &
wait
python inference_global.py --config config_global.yaml --experiment daily_global_SNOW_inference &
python inference_global.py --config config_global.yaml --experiment daily_global_TWS_inference &
python inference_global.py --config config_global.yaml --experiment daily_global_LZS_inference &
python inference_global.py --config config_global.yaml --experiment daily_global_AGW_inference &
python inference_global.py --config config_global.yaml --experiment daily_global_streamflow_inference &
wait
python analysis_global.py --config config_global.yaml --experiment daily_global_PET_analysis  &
python analysis_global.py --config config_global.yaml --experiment daily_global_ET_analysis &
python analysis_global.py --config config_global.yaml --experiment daily_global_SUPY_analysis &
python analysis_global.py --config config_global.yaml --experiment daily_global_WYIE_analysis &
wait
python analysis_global.py --config config_global.yaml --experiment daily_global_SNOW_analysis &
python analysis_global.py --config config_global.yaml --experiment daily_global_TWS_analysis &
python analysis_global.py --config config_global.yaml --experiment daily_global_LZS_analysis &
python analysis_global.py --config config_global.yaml --experiment daily_global_AGW_analysis &
python analysis_global.py --config config_global.yaml --experiment daily_global_streamflow_analysis &
wait



# # Hourly global streamflow with daily IMV inputs
# python combine_daily_imv_outputs.py
# python combine_daily_imv_outputs_hourly_streamflow_hourly_eddy.py

# python train_global.py --config config_global.yaml --experiment hourly_global_streamflow_no_IMVs &
# python train_global.py --config config_global.yaml --experiment hourly_global_streamflow_observed_IMVs &
# python train_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_PET &
# python train_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_ET &
# python train_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_SUPY &
# wait
# python train_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_WYIE &
# python train_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_SNOW &
# python train_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_TWS &
# python train_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_LZS &
# python train_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_AGW &
# python train_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_streamflow &
# wait

# python inference_global.py --config config_global.yaml --experiment hourly_global_streamflow_no_IMVs_inference &
# python inference_global.py --config config_global.yaml --experiment hourly_global_streamflow_observed_IMVs_inference &
# python inference_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_PET_inference &
# python inference_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_ET_inference &
# python inference_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_SUPY_inference &
# wait
# python inference_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_WYIE_inference &
# python inference_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_SNOW_inference &
# python inference_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_TWS_inference &
# python inference_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_LZS_inference &
# python inference_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_AGW_inference &
# python inference_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_streamflow_inference &
# wait

# python analysis_global.py --config config_global.yaml --experiment hourly_global_streamflow_no_IMVs_analysis &
# python analysis_global.py --config config_global.yaml --experiment hourly_global_streamflow_observed_IMVs_analysis &
# python analysis_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_PET_analysis &
# python analysis_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_ET_analysis &
# python analysis_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_SUPY_analysis &
# wait
# python analysis_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_WYIE_analysis &
# python analysis_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_SNOW_analysis &
# python analysis_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_TWS_analysis &
# python analysis_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_LZS_analysis &
# python analysis_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_AGW_analysis &
# python analysis_global.py --config config_global.yaml --experiment hourly_global_streamflow_pred_streamflow_analysis &


