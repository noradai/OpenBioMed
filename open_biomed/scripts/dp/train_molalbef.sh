#!/bin/bash

# BBBP, Tox21, Toxcast, sider, clintox, muv, hiv, bace
dataset="BBBP"

python tasks/mol_task/dp.py \
--device cuda:0 \
--dataset MoleculeNet \
--dataset_path ../datasets/dp/moleculenet \
--dataset_name $dataset \
--config_path ./configs/dp/molalbef.json \
--output_path ../ckpts/finetune_ckpts/dp/molalbef_finetune.pth \
--num_workers 1 \
--mode train \
--batch_size 32 \
--epochs 80 \
--patience 20 \
--seed 2