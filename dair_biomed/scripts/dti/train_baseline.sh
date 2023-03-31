#!/bin/bash
MODE="train"
MODEL="mgraphdta"
DEVICE="cuda:0"
EPOCHS=300

python tasks/mol_task/dti.py \
--device ${DEVICE} \
--config_path ./configs/dti/${MODEL}.json \
--dataset yamanishi08 \
--dataset_path ../datasets/dti/Yamanishi08 \
--output_path ../ckpts/finetune_ckpts/dti/${MODEL}.pth \
--mode kfold \
--epochs ${EPOCHS} \
--num_workers 4 \
--batch_size 128 \
--lr 1e-3 \
--logging_steps 50 \
--patience 20