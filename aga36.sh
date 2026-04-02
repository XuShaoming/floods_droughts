# python process_flow_data.py

# python process_eddev_data.py --all --basin "KettleR_Watersheds" --scenario "Historical"
# python combine_eddev_flow.py --basin "KettleRiverModels" --scenario "hist_scaled"

# python process_eddev_data.py --all --basin "KettleR_Watersheds" --scenario "Historical"
# python process_eddev_data.py --all --basin "KettleR_Watersheds" --scenario "RCP4.5"
# python process_eddev_data.py --all --basin "KettleR_Watersheds" --scenario "RCP8.5"

# python combine_eddev_flow.py --basin "KettleRiverModels" --scenario "hist_scaled"
# python combine_eddev_flow.py --basin "KettleRiverModels" --scenario "RCP4.5"
# python combine_eddev_flow.py --basin "KettleRiverModels" --scenario "RCP8.5"


# python process_eddev_data.py --all --basin "WatonwanR_Watersheds" --scenario "Historical"
# python process_eddev_data.py --all --basin "LeSueurR_Watersheds" --scenario "Historical"
# python process_eddev_data.py --all --basin "BlueEarthR_Watersheds" --scenario "Historical"



# python combine_eddev_flow.py --basin "BlueEarth" --scenario "hist_scaled"
# python combine_eddev_flow.py --basin "LeSueur" --scenario "hist_scaled"
# python combine_eddev_flow.py --basin "Watonwan" --scenario "hist_scaled"

#######


# # python process_eddev_data.py --all --basin "WatonwanR_Watersheds" --scenario "RCP4.5"
# python process_eddev_data.py --all --basin "WatonwanR_Watersheds" --scenario "RCP8.5"
# python process_eddev_data.py --all --basin "KettleR_Watersheds" --scenario "RCP4.5"
# python process_eddev_data.py --all --basin "KettleR_Watersheds" --scenario "RCP8.5"
# python combine_eddev_flow.py --basin "Watonwan" --scenario "RCP4.5"
# python combine_eddev_flow.py --basin "Watonwan" --scenario "RCP8.5"
# python combine_eddev_flow.py --basin "KettleRiverModels" --scenario "RCP4.5"
# python combine_eddev_flow.py --basin "KettleRiverModels" --scenario "RCP8.5"




# # python inference.py --model-dir experiments/streamflow_exp2 --model-trained final_model.pth --dataset train --analysis &
# python inference.py --model-dir experiments/streamflow_exp2 --model-trained final_model.pth --dataset test --analysis & 
# python inference.py --model-dir experiments/streamflow_exp2 --model-trained final_model.pth --dataset val --analysis &

# python inference.py --model-dir experiments/streamflow_exp2 --model-trained best_model.pth --dataset train --analysis &
# python inference.py --model-dir experiments/streamflow_exp2 --model-trained best_model.pth --dataset val --analysis &
# python inference.py --model-dir experiments/streamflow_exp2 --model-trained best_model.pth --dataset test --analysis &


# python inference.py --model-dir experiments/streamflow_exp2 --model-trained best_model.pth --dataset val --analysis


# python train_global.py --config config_global.yaml --experiment streamflow_global_exp1_test
# python train_global.py --config config_global.yaml --experiment streamflow_global_exp1


# python inference_global.py --config config_global.yaml --experiment streamflow_global_exp1_inference
# # python analysis_global.py --config config_global.yaml --experiment streamflow_global_exp1_inference
# python analysis_global.py --config config_global.yaml --experiment streamflow_global_exp1_inference_train
# python analysis_global.py --config config_global.yaml --experiment streamflow_global_exp1_inference_val
# # python inference_global.py --model-dir <exp_dir> --dataset test


# {
# # python inference.py --model-dir experiments/streamflow_exp1 --model-trained best_model.pth --dataset test &
# # python inference.py --model-dir experiments/streamflow_exp1 --model-trained best_model.pth --dataset val &
# # python inference.py --model-dir experiments/streamflow_exp1 --model-trained best_model.pth --dataset train &
# }
# {
# python inference.py --model-dir experiments/streamflow_BlueEarth_hist_scaled_exp1 --model-trained best_model.pth --dataset test &
# python inference.py --model-dir experiments/streamflow_BlueEarth_hist_scaled_exp1 --model-trained best_model.pth --dataset val &
# python inference.py --model-dir experiments/streamflow_BlueEarth_hist_scaled_exp1 --model-trained best_model.pth --dataset train &
# wait
# }
# {
# python analysis.py --model-dir experiments/streamflow_BlueEarth_hist_scaled_exp1 --model-trained best_model.pth --dataset test &
# python analysis.py --model-dir experiments/streamflow_BlueEarth_hist_scaled_exp1 --model-trained best_model.pth --dataset val & 
# python analysis.py --model-dir experiments/streamflow_BlueEarth_hist_scaled_exp1 --model-trained best_model.pth --dataset train &
# wait
# }

# {
# python inference.py --model-dir experiments/streamflow_LeSueur_hist_scaled_exp1 --model-trained best_model.pth --dataset test &
# python inference.py --model-dir experiments/streamflow_LeSueur_hist_scaled_exp1 --model-trained best_model.pth --dataset val &
# python inference.py --model-dir experiments/streamflow_LeSueur_hist_scaled_exp1 --model-trained best_model.pth --dataset train &
# wait
# }
# {
# python analysis.py --model-dir experiments/streamflow_LeSueur_hist_scaled_exp1 --model-trained best_model.pth --dataset test &
# python analysis.py --model-dir experiments/streamflow_LeSueur_hist_scaled_exp1 --model-trained best_model.pth --dataset val & 
# python analysis.py --model-dir experiments/streamflow_LeSueur_hist_scaled_exp1 --model-trained best_model.pth --dataset train &
# wait
# }

# {
# python inference.py --model-dir experiments/streamflow_Watonwan_hist_scaled_exp1 --model-trained best_model.pth --dataset test &
# python inference.py --model-dir experiments/streamflow_Watonwan_hist_scaled_exp1 --model-trained best_model.pth --dataset val &
# python inference.py --model-dir experiments/streamflow_Watonwan_hist_scaled_exp1 --model-trained best_model.pth --dataset train &
# wait
# }
# {
# python analysis.py --model-dir experiments/streamflow_Watonwan_hist_scaled_exp1 --model-trained best_model.pth --dataset test &
# python analysis.py --model-dir experiments/streamflow_Watonwan_hist_scaled_exp1 --model-trained best_model.pth --dataset val & 
# python analysis.py --model-dir experiments/streamflow_Watonwan_hist_scaled_exp1 --model-trained best_model.pth --dataset train &
# wait
# }

# {
# python train_hmtl_global.py --config config_local.yaml --experiment streamflow_hmtl_kettle_river --seed 42 &
# wait
# }

# {
# python inference_hmtl_global.py --config config_local.yaml --experiment streamflow_hmtl_kettle_river_inference &
# python inference_hmtl_global.py --config config_local.yaml --experiment streamflow_hmtl_kettle_river_inference_val &
# python inference_hmtl_global.py --config config_local.yaml --experiment streamflow_hmtl_kettle_river_inference_train &
# wait
# }

# {
# python analysis_hmtl_global.py --config config_local.yaml --experiment streamflow_hmtl_kettle_river_analysis --split test val train &
# wait
# }

# {
# # python process_eddev_data.py --all --basin "LittleFork_Watersheds" --scenario "Historical" &
# # python process_eddev_data.py --all --basin "SnakeSE_Watersheds" --scenario "Historical" & 
# # python process_eddev_data.py --all --basin "Zumbro_Watersheds" --scenario "Historical" &
# }


# python combine_eddev_flow.py --basin "KettleRiverModels" --scenario "hist_scaled"
# python combine_eddev_flow.py --basin "KettleRiverModels" --scenario "RCP4.5"
# python combine_eddev_flow.py --basin "KettleRiverModels" --scenario "RCP8.5"

# for basin in Watonwan LeSueur KettleRiverModels BlueEarth LittleFork Zumbro SnakeSE; do
#     python combine_eddev_flow.py --basin $basin --scenario "hist_scaled" --resolution "daily" &
#     python combine_eddev_flow.py --basin $basin --scenario "RCP4.5" --resolution "daily" &
#     python combine_eddev_flow.py --basin $basin --scenario "RCP8.5" --resolution "daily" &
#     wait
# done


# python process_eddev_data.py --all --basin "SnakeSE_Watersheds" --scenario "RCP4.5" &
# python process_eddev_data.py --all --basin "SnakeSE_Watersheds" --scenario "RCP8.5" & 
# wait
# python process_eddev_data.py --all --basin "LittleFork_Watersheds" --scenario "RCP4.5" & 
# python process_eddev_data.py --all --basin "LittleFork_Watersheds" --scenario "RCP8.5" &
# wait
# python process_eddev_data.py --all --basin "Zumbro_Watersheds" --scenario "RCP4.5" & 
# python process_eddev_data.py --all --basin "Zumbro_Watersheds" --scenario "RCP8.5" &
# wait

# python process_flow_data.py

# for basin in LittleFork Zumbro SnakeSE; do
#     python combine_eddev_flow.py --basin $basin --scenario "hist_scaled" --resolution "daily" &
#     python combine_eddev_flow.py --basin $basin --scenario "RCP4.5" --resolution "daily" &
#     python combine_eddev_flow.py --basin $basin --scenario "RCP8.5" --resolution "daily" &
#     wait
# done


# {
# # python train_hmtl_global.py --experiment streamflow_hmtl_global
# # wait
# # python inference_hmtl_global.py --experiment streamflow_hmtl_global_inference &
# python inference_hmtl_global.py --experiment streamflow_hmtl_global_inference_val &
# python inference_hmtl_global.py --experiment streamflow_hmtl_global_inference_train &
# wait 
# python analysis_hmtl_global.py --experiment streamflow_hmtl_global_analysis
# wait
# }



# python combined_eddev_flow_daily.py --basin KettleRiverModels --scenario hist_scaled


# for basin in Watonwan LeSueur KettleRiverModels BlueEarth LittleFork Zumbro SnakeSE; do
#     python combined_eddev_flow_daily.py --basin $basin --scenario hist_scaled &
#     python combined_eddev_flow_daily.py --basin $basin --scenario RCP4.5 &
#     python combined_eddev_flow_daily.py --basin $basin --scenario RCP8.5 &
#     wait
# done

# python train_global.py --config config_global.yaml --experiment daily_global_streamflow
# python inference_global.py --config config_global.yaml --experiment daily_global_streamflow_inference
# python analysis_global.py --config config_global.yaml --experiment daily_global_streamflow_analysis

# python combined_eddev_flow_daily.py --basin BlueEarth --scenario hist_scaled &
# python combined_eddev_flow_daily.py --basin KettleRiverModels --scenario hist_scaled &



# for exp in daily_global_PET daily_global_ET daily_global_SUPY daily_global_WYIE daily_global_SNOW daily_global_TWS daily_global_LZS daily_global_AGW daily_global_streamflow; do
#     python train_global.py --config config_global.yaml --experiment $exp
#     python inference_global.py --config config_global.yaml --experiment ${exp}_inference
#     python analysis_global.py --config config_global.yaml --experiment ${exp}_analysis
# done

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
# python inference_global.py --config config_global.yaml --experiment daily_global_PET_inference & 
# python inference_global.py --config config_global.yaml --experiment daily_global_ET_inference &
# python inference_global.py --config config_global.yaml --experiment daily_global_SUPY_inference &
# python inference_global.py --config config_global.yaml --experiment daily_global_WYIE_inference &
# wait
# python inference_global.py --config config_global.yaml --experiment daily_global_SNOW_inference &
# python inference_global.py --config config_global.yaml --experiment daily_global_TWS_inference &
# python inference_global.py --config config_global.yaml --experiment daily_global_LZS_inference &
# python inference_global.py --config config_global.yaml --experiment daily_global_AGW_inference &
# wait
# python analysis_global.py --config config_global.yaml --experiment daily_global_PET_analysis  &
# python analysis_global.py --config config_global.yaml --experiment daily_global_ET_analysis &
# python analysis_global.py --config config_global.yaml --experiment daily_global_SUPY_analysis &
# python analysis_global.py --config config_global.yaml --experiment daily_global_WYIE_analysis &
# wait
# python analysis_global.py --config config_global.yaml --experiment daily_global_SNOW_analysis &
# python analysis_global.py --config config_global.yaml --experiment daily_global_TWS_analysis &
# python analysis_global.py --config config_global.yaml --experiment daily_global_LZS_analysis &
# python analysis_global.py --config config_global.yaml --experiment daily_global_AGW_analysis &
# wait

# python -m pdb combined_eddev_flow_daily.py --basin KettleRiverModels --scenario hist_scaled



## Mar 17, 14:06, 2025
# python process_flow_data.py
# combine_eddev_flow.py for KettleRiverModels
# python combined_eddev_flow_daily.py --basin $basin --scenario hist_scaled & for KettleRiverModels
# All daily_global_* experiments, and streamflow_hmtl_global experiment.




# python process_flow_data.py
# wait
# python combine_eddev_flow.py --basin KettleRiverModels --scenario "hist_scaled" --resolution "daily" &
# python combine_eddev_flow.py --basin KettleRiverModels --scenario "RCP4.5" --resolution "daily" &
# python combine_eddev_flow.py --basin KettleRiverModels --scenario "RCP8.5" --resolution "daily" &
# wait
# python combined_eddev_flow_daily.py --basin KettleRiverModels --scenario hist_scaled &
# python combined_eddev_flow_daily.py --basin KettleRiverModels --scenario RCP4.5 &
# python combined_eddev_flow_daily.py --basin KettleRiverModels --scenario RCP8.5 &
# wait

# python train_global.py --config config_global.yaml --experiment daily_global_streamflow &    
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
# python inference_global.py --config config_global.yaml --experiment daily_global_streamflow_inference &
# python inference_global.py --config config_global.yaml --experiment daily_global_PET_inference & 
# python inference_global.py --config config_global.yaml --experiment daily_global_ET_inference &
# python inference_global.py --config config_global.yaml --experiment daily_global_SUPY_inference &
# python inference_global.py --config config_global.yaml --experiment daily_global_WYIE_inference &
# wait
# python inference_global.py --config config_global.yaml --experiment daily_global_SNOW_inference &
# python inference_global.py --config config_global.yaml --experiment daily_global_TWS_inference &
# python inference_global.py --config config_global.yaml --experiment daily_global_LZS_inference &
# python inference_global.py --config config_global.yaml --experiment daily_global_AGW_inference &
# wait
# python analysis_global.py --config config_global.yaml --experiment daily_global_streamflow_analysis &
# python analysis_global.py --config config_global.yaml --experiment daily_global_PET_analysis  &
# python analysis_global.py --config config_global.yaml --experiment daily_global_ET_analysis &
# python analysis_global.py --config config_global.yaml --experiment daily_global_SUPY_analysis &
# python analysis_global.py --config config_global.yaml --experiment daily_global_WYIE_analysis &
# wait
# python analysis_global.py --config config_global.yaml --experiment daily_global_SNOW_analysis &
# python analysis_global.py --config config_global.yaml --experiment daily_global_TWS_analysis &
# python analysis_global.py --config config_global.yaml --experiment daily_global_LZS_analysis &
# python analysis_global.py --config config_global.yaml --experiment daily_global_AGW_analysis &
# wait
# python train_hmtl_global.py --experiment streamflow_hmtl_global
# wait
# python inference_hmtl_global.py --experiment streamflow_hmtl_global_inference &
# python inference_hmtl_global.py --experiment streamflow_hmtl_global_inference_val &
# python inference_hmtl_global.py --experiment streamflow_hmtl_global_inference_train &
# wait 
# python analysis_hmtl_global.py --experiment streamflow_hmtl_global_analysis
# wait


# python analysis_global.py --config config_global.yaml --experiment daily_global_streamflow_analysis &
# python analysis_global.py --config config_global.yaml --experiment daily_global_PET_analysis  &
# python analysis_global.py --config config_global.yaml --experiment daily_global_ET_analysis &
# python analysis_global.py --config config_global.yaml --experiment daily_global_SUPY_analysis &
# python analysis_global.py --config config_global.yaml --experiment daily_global_WYIE_analysis &
# wait
# python analysis_hmtl_global.py --experiment streamflow_hmtl_global_analysis &
# python analysis_global.py --config config_global.yaml --experiment daily_global_SNOW_analysis &
# python analysis_global.py --config config_global.yaml --experiment daily_global_TWS_analysis &
# python analysis_global.py --config config_global.yaml --experiment daily_global_LZS_analysis &
# python analysis_global.py --config config_global.yaml --experiment daily_global_AGW_analysis &


# Hourly global streamflow with daily IMV inputs
# python combine_imv_outputs.py
# python combine_daily_imv_outputs_hourly_streamflow_hourly_eddy.py
# python train_global.py --config config_global.yaml --experiment hourly_global_streamflow
# python inference_global.py --config config_global.yaml --experiment hourly_global_streamflow_inference
# python analysis_global.py --config config_global.yaml --experiment hourly_global_streamflow_analysis



# python train_global.py --config config_global.yaml --experiment hourly_global_streamflow_no_IMVs &
# python train_global.py --config config_global.yaml --experiment hourly_global_streamflow_observed_IMVs &
# wait
# python inference_global.py --config config_global.yaml --experiment hourly_global_streamflow_no_IMVs_inference &
# python inference_global.py --config config_global.yaml --experiment hourly_global_streamflow_observed_IMVs_inference &

python analysis_global.py --config config_global.yaml --experiment hourly_global_streamflow_no_IMVs_analysis &
python analysis_global.py --config config_global.yaml --experiment hourly_global_streamflow_observed_IMVs_analysis &