#!/bin/bash
#SBATCH --job-name open-dair-biomed
#SBATCH --array 0
#SBATCH --gres gpu:a100:1
#SBATCH --time 12:00:00 
#SBATCH --output ../logs/momu_zero_shot.log
python tasks/multi_modal_task/mtr.py \
--device cuda:1 \
--dataset PCdes \
--dataset_path ../datasets/mtr/PCdes \
--config_path ./configs/mtr/momu.json \
--init_checkpoint ../ckpts/fusion_ckpts/momu/MoMu-K.ckpt \
--num_workers 1 \
--mode train \
--epochs 100 \
--patience 10