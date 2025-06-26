# python train.py --config 'config.yaml' --experiment streamflow_exp1 --seed 42

# python inference.py --model-dir experiments/streamflow_exp1 --model-trained best_model.pth --dataset test --analysis

# python inference.py --model-dir experiments/streamflow_exp1 --model-trained best_model.pth --dataset train --analysis

python inference.py --model-dir experiments/streamflow_exp1 --model-trained final_model.pth --dataset train --analysis
python inference.py --model-dir experiments/streamflow_exp1 --model-trained final_model.pth --dataset test --analysis
python inference.py --model-dir experiments/streamflow_exp1 --model-trained final_model.pth --dataset val --analysis

# python inference.py --model-dir experiments/streamflow_exp1 --model-trained best_model.pth --dataset train --analysis
# python inference.py --model-dir experiments/streamflow_exp1 --model-trained best_model.pth --dataset val --analysis
# python inference.py --model-dir experiments/streamflow_exp1 --model-trained best_model.pth --dataset test --analysis



