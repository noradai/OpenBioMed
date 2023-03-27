#!/bin/bash
MODE="zero_shot"
TASK_MODE="paragraph"
MODEL="molalbef"
DEVICE="cuda:0"
EPOCHS=100

CKPT="None"
PARAM_KEY="None"

if [ $MODEL = "molalbef" ]; 
then
    CKPT="../ckpts/fusion_ckpts/molalbef-gc-mlm/checkpoint_449.pth"
    PARAM_KEY="model"
elif [ $MODEL = "momu" ]; 
then
    CKPT="../ckpts/fusion_ckpts/momu/MoMu-K.ckpt"
    PARAM_KEY="state_dict"
fi

python tasks/multi_modal_task/mtr.py \
--device ${DEVICE} \
--dataset PCdes \
--dataset_path ../datasets/mtr/PCdes \
--dataset_mode ${TASK_MODE} \
--filter \
--filter_path /share/project/molpretrain/data/pair.txt \
--config_path ./configs/mtr/${MODEL}.json \
--init_checkpoint ${CKPT} \
--output_path ../ckpts/finetune_ckpts/${MODEL}-${TASK_MODE}-finetune.pth \
--param_key ${PARAM_KEY} \
--num_workers 1 \
--mode ${MODE} \
--patience 20 \
--epochs ${EPOCHS} \
--rerank_num 32