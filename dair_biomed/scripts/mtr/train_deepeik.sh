#!/bin/bash
python tasks/multi_modal_task/mtr.py \
--device cuda:1 \
--dataset PCdes \
--dataset_path ../datasets/mtr/PCdes \
--config_path ./configs/mtr/deepeik.json \
--num_workers 1 \
--mode zero_shot \
--epochs 100 \
--patience 10