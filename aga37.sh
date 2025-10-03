

# python inference.py --model-dir experiments/streamflow_exp1 --model-trained best_model.pth --dataset test --analysis
# python inference.py --model-dir experiments/streamflow_exp1 --model-trained best_model.pth --dataset train --analysis

# python inference.py --model-dir experiments/streamflow_exp1 --model-trained final_model.pth --dataset train --analysis
# python inference.py --model-dir experiments/streamflow_exp1 --model-trained final_model.pth --dataset test --analysis
# python inference.py --model-dir experiments/streamflow_exp1 --model-trained final_model.pth --dataset val --analysis

# python inference.py --model-dir experiments/streamflow_exp1 --model-trained best_model.pth --dataset train --analysis
# python inference.py --model-dir experiments/streamflow_exp1 --model-trained best_model.pth --dataset val --analysis
# python inference.py --model-dir experiments/streamflow_exp1 --model-trained best_model.pth --dataset test --analysis



# python process_eddev_data.py --all --basin "LeSueurR_Watersheds" --scenario "RCP4.5"
# python process_eddev_data.py --all --basin "LeSueurR_Watersheds" --scenario "RCP8.5"
# python process_eddev_data.py --all --basin "BlueEarthR_Watersheds" --scenario "RCP4.5"
# python process_eddev_data.py --all --basin "BlueEarthR_Watersheds" --scenario "RCP8.5"

# python combine_eddev_flow.py --basin "LeSueur" --scenario "RCP4.5"
# python combine_eddev_flow.py --basin "LeSueur" --scenario "RCP8.5"
# python combine_eddev_flow.py --basin "BlueEarth" --scenario "RCP4.5"
# python combine_eddev_flow.py --basin "BlueEarth" --scenario "RCP8.5"
# module load gcc/11.2.0

# python train.py --config 'config.yaml' --experiment streamflow_exp1 --seed 42 &
# python train.py --config 'config.yaml' --experiment streamflow_exp2 --seed 42 &



# python inference.py --model-dir experiments/streamflow_exp1 --model-trained final_model.pth --dataset train --analysis &
# python inference.py --model-dir experiments/streamflow_exp1 --model-trained final_model.pth --dataset test --analysis & 
# python inference.py --model-dir experiments/streamflow_exp1 --model-trained final_model.pth --dataset val --analysis &

# python inference.py --model-dir experiments/streamflow_exp1 --model-trained best_model.pth --dataset train --analysis &
# python inference.py --model-dir experiments/streamflow_exp1 --model-trained best_model.pth --dataset val --analysis &
# python inference.py --model-dir experiments/streamflow_exp1 --model-trained best_model.pth --dataset test --analysis &

# python train_hmtl.py --config 'config.yaml' --experiment streamflow_hmtl_test --seed 42
# python train_hstl.py --config 'config.yaml' --experiment streamflow_hstl --seed 42



python inference_mtl.py --model-dir experiments/streamflow_hmtl --model-trained best_model.pth --dataset train --analysis &
python inference_mtl.py --model-dir experiments/streamflow_hmtl --model-trained best_model.pth --dataset val --analysis &
python inference_mtl.py --model-dir experiments/streamflow_hmtl --model-trained best_model.pth --dataset test --analysis &


# python inference_mtl.py --model-dir experiments/streamflow_hmtl --model-trained best_model.pth --dataset train --analysis &
# python inference_mtl.py --model-dir experiments/streamflow_hmtl --model-trained best_model.pth --dataset val --analysis &
# # python inference_mtl.py --model-dir experiments/streamflow_hstl --model-trained best_model.pth --dataset test --analysis &


(
# python train.py --config 'config.yaml' --experiment streamflow_exp3 --seed 42 &
# python train.py --config 'config.yaml' --experiment streamflow_exp4 --seed 42 &
# python train.py --config 'config.yaml' --experiment streamflow_exp5 --seed 42 &
# python train.py --config 'config.yaml' --experiment streamflow_exp6 --seed 42 &
wait
)

(
# python inference.py --model-dir experiments/streamflow_exp3 --model-trained final_model.pth --dataset train --analysis &
# python inference.py --model-dir experiments/streamflow_exp3 --model-trained final_model.pth --dataset test --analysis & 
# python inference.py --model-dir experiments/streamflow_exp3 --model-trained final_model.pth --dataset val --analysis &

# python inference.py --model-dir experiments/streamflow_exp4 --model-trained final_model.pth --dataset train --analysis &
# python inference.py --model-dir experiments/streamflow_exp4 --model-trained final_model.pth --dataset test --analysis & 
# python inference.py --model-dir experiments/streamflow_exp4 --model-trained final_model.pth --dataset val --analysis &

# python inference.py --model-dir experiments/streamflow_exp5 --model-trained final_model.pth --dataset train --analysis &
# python inference.py --model-dir experiments/streamflow_exp5 --model-trained final_model.pth --dataset test --analysis & 
# python inference.py --model-dir experiments/streamflow_exp5 --model-trained final_model.pth --dataset val --analysis &

# python inference.py --model-dir experiments/streamflow_exp6 --model-trained final_model.pth --dataset train --analysis &
# python inference.py --model-dir experiments/streamflow_exp6 --model-trained final_model.pth --dataset test --analysis & 
# python inference.py --model-dir experiments/streamflow_exp6 --model-trained final_model.pth --dataset val --analysis &
wait
)