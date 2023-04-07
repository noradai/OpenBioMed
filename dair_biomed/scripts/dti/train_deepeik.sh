#!/bin/bash
MODE="train"
MODEL="deepeik"
BASE="mgraphdta"
DEVICE=1
EPOCHS=500

CUDA_VISIBLE_DEVICES=${DEVICE} python tasks/mol_task/dti.py \
--device cuda:0 \
--config_path ./configs/dti/${MODEL}-${BASE}.json \
--dataset yamanishi08 \
--dataset_path ../datasets/dti/Yamanishi08 \
--output_path ../ckpts/finetune_ckpts/dti/${MODEL}.pth \
--mode kfold \
--epochs ${EPOCHS} \
--num_workers 4 \
--batch_size 128 \
--lr 1e-3 \
--logging_steps 50 \
--patience 200