#!/bin/bash
#SBATCH --job-name open-dair-biomed
#SBATCH --array 0
#SBATCH --gres gpu:a100:1
#SBATCH --time 12:00:00 
#SBATCH --output ../logs/momu_zero_shot.log
MODEL="molalbef"
MOL="graphmvp"

python tasks/multi_modal_task/mtr.py \
--device cuda:0 \
--dataset PCdes \
--dataset_path ../datasets/mtr/PCdes \
--config_path ./configs/mtr/$MODEL.json \
--init_checkpoint ../ckpts/fusion_ckpts/${MODEL}-${MOL}/checkpoint_299.pth \
--output_path ../ckpts/finetune_ckpts/${MODEL}-${MOL}-finetune.pth \
--param_key model \
--num_workers 1 \
--mode train \
--epochs 100 \
--patience 10