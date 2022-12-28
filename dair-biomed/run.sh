#!/bin/bash
#SBATCH --job-name dti
#SBATCH --array 0
#SBATCH --gres gpu:a100:1
#SBATCH --time 1-00:00:00 
#SBATCH --output save/bmkg_kg_text.log

log_dir=train_08

CUDA_VISIBLE_DEVICES=0 python train.py > ${log_dir}.log 2>&1