#!/bin/bash
MODE="train"
MODEL="deepdta"
DEVICE="cuda:0"
EPOCHS=300

python tasks/mol_task/dti.py \
--device ${DEVICE} \
--config_path ./configs/dti/${MODEL}.json \
--dataset yamanishi08 \
--dataset_path ../datasets/mti/Yamanishi08 \
--output_path ../ckpts/finetune_ckpts/dti/ \
--mode kfold \
--epochs ${EPOCHS} \
--num_workers 4 \
--batch_size 512 \
--logging_steps 300 \
--patience 20