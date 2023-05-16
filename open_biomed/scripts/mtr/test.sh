#!/bin/bash
MODE="zero_shot"
TASK_MODE="paragraph"
MODEL="molalbef"
DEVICE="cuda:0"
EPOCHS=100

CKPT="None"
PARAM_KEY="model_state_dict"
RERANK="no_rerank"

FILTER_FILE="../datasets/mtr/momu_pretrain/pair.txt"

if [ $MODEL = "molalbef" ]; 
then
    CKPT="../ckpts/finetune_ckpts/molalbef-paragraph-finetune.pth"
    RERANK="no_rerank"
elif [ $MODEL = "momu" ]; 
then
    CKPT="../ckpts/finetune_ckpts/momu-paragraph-finetune.pth"
    RERANK="no_rerank"
fi

python tasks/multi_modal_task/mtr.py \
--device ${DEVICE} \
--dataset PCdes \
--dataset_path ../datasets/mtr/PCdes \
--dataset_mode ${TASK_MODE} \
--filter \
--filter_path ${FILTER_FILE} \
--config_path ./configs/mtr/${MODEL}.json \
--init_checkpoint ${CKPT} \
--output_path ../ckpts/finetune_ckpts/${MODEL}-${TASK_MODE}-finetune.pth \
--param_key ${PARAM_KEY} \
--num_workers 1 \
--mode ${MODE} \
--patience 20 \
--epochs ${EPOCHS} \
--${RERANK} \
--rerank_num 32 \
--alpha_m2t 0.9 \
--alpha_t2m 0.8