# python process_flow_data.py

# python process_eddev_data.py --all --basin "KettleR_Watersheds" --scenario "Historical"
# python combine_eddev_flow.py --basin "KettleRiverModels" --scenario "hist_scaled"

# python process_eddev_data.py --all --basin "KettleR_Watersheds" --scenario "Historical"
# python process_eddev_data.py --all --basin "KettleR_Watersheds" --scenario "RCP4.5"
# python process_eddev_data.py --all --basin "KettleR_Watersheds" --scenario "RCP8.5"


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
# python inference.py --model-dir experiments/streamflow_exp1 --model-trained best_model.pth --dataset test &
# python inference.py --model-dir experiments/streamflow_exp1 --model-trained best_model.pth --dataset val &
# python inference.py --model-dir experiments/streamflow_exp1 --model-trained best_model.pth --dataset train &
# wait
# }


python analysis.py --model-dir experiments/streamflow_exp1 --model-trained best_model.pth --dataset test &
python analysis.py --model-dir experiments/streamflow_exp1 --model-trained best_model.pth --dataset val & 
python analysis.py --model-dir experiments/streamflow_exp1 --model-trained best_model.pth --dataset train &