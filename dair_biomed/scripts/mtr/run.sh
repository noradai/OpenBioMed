#!/bin/bash
MODE="zero_shot"
TASK_MODE="paragraph"
MODEL="momu"
DEVICE="cuda:0"
EPOCHS=100

CKPT=""
PARAM_KEY=""
RERANK="no_rerank"

FILTER_FILE="../datasets/mtr/momu_pretrain/pair.txt"

if [ $MODEL = "molalbef" ]; 
then
    CKPT="../ckpts/fusion_ckpts/molalbef/molalbef-ke.pth"
    PARAM_KEY="model"
    RERANK="rerank"
elif [ $MODEL = "momu" ]; 
then
    CKPT="../ckpts/fusion_ckpts/momu/MoMu-K.ckpt"
    PARAM_KEY="state_dict"
    RERANK="no_rerank"
elif [ $MODEL = "biomedgpt" ];
then
    CKPT="../ckpts/fusion_ckpts/biomedgpt/checkpoint_99.pth"
    param_key="None"
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