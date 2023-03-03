#!/bin/bash
python tasks/multi_modal_task/mtr.py \
--device cuda:0 \
--dataset PCdes \
--dataset_path ../datasets/mtr/PCdes \
--config_path ./configs/mtr/deepeik.json \
--num_workers 1 \
--mode train \
--epochs 100 \
--patience 10