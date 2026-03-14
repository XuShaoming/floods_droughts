for f in \
process_flow_data_hourly.py \
config_local.yaml \
inference.py \
dataloader.py \
config.yaml \
analysis.py \
train.py \
inference_mtl.py \
train_hmtl_uncertainty.py \
inference_mtl_cmb.py \
train_hmtl_cmb.py \
exp_compare.py \
train_hstl.py \
train_hmtl.py \
dataloader_hmtl.py \
combined_visualize.py
do
  if git cat-file -e origin/main:"$f" 2>/dev/null; then
    echo "[IN origin/main] $f"
  else
    echo "[NOT in origin/main] $f"
  fi
done