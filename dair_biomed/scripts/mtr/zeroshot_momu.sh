#!/bin/bash
#SBATCH --job-name open-dair-biomed
#SBATCH --array 0
#SBATCH --gres gpu:a100:1
#SBATCH --time 12:00:00 
#SBATCH --output ../logs/momu_zero_shot.log
MODE="sentence"

python tasks/multi_modal_task/mtr.py \
--device cuda:0 \
--dataset PCdes \
--dataset_path ../datasets/mtr/PCdes \
--dataset_mode ${MODE} \
--config_path ./configs/mtr/momu.json \
--init_checkpoint ../ckpts/fusion_ckpts/momu/MoMu-K.ckpt \
--param_key state_dict \
--num_workers 1 \
--mode zero_shot