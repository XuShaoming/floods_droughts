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


{
python train_hmtl_global.py --experiment streamflow_hmtl_global
wait
python inference_hmtl_global.py --experiment streamflow_hmtl_global_inference &
python inference_hmtl_global.py --experiment streamflow_hmtl_global_inference_val &
python inference_hmtl_global.py --experiment streamflow_hmtl_global_inference_train &
wait 
python analysis_hmtl_global.py --experiment streamflow_hmtl_global_analysis --target-group final
wait
}

