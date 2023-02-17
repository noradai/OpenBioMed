#!/bin/bash
#SBATCH --job-name dti
#SBATCH --array 0
#SBATCH --gres gpu:a100:1
#SBATCH --time 1-00:00:00 
#SBATCH --output save/bmkg_kg_text.log
# ps aux | grep "train.py" | awk '{print $2}'  | xargs kill -9

# log_dir=yamanish08
# CUDA_VISIBLE_DEVICES=0 python train.py --config-path ./configs/yamanish08.json > ${log_dir}.log 2>&1

# log_dir=davis
# CUDA_VISIBLE_DEVICES=0 python train.py --config-path ./configs/davis.json > ${log_dir}.log 2>&1

log_dir=kiba
CUDA_VISIBLE_DEVICES=0 python train.py --config-path ./configs/kiba.json > ${log_dir}.log 2>&1