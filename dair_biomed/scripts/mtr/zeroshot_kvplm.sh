#!/bin/bash
#SBATCH --job-name open-dair-biomed
#SBATCH --array 0
#SBATCH --gres gpu:a100:1
#SBATCH --time 12:00:00 
#SBATCH --output ../logs/kvplm_zero_shot.log
MODEL="kvplm"

python tasks/multi_modal_task/mtr.py \
--device cuda:0 \
--dataset PCdes \
--dataset_path ../datasets/mtr/PCdes \
--config_path ./configs/mtr/${MODEL}.json \
--num_workers 1 \
--mode zero_shot