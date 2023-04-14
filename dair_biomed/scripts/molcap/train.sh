#!/bin/bash
MODE="traintest"
MODEL="molalbef"
DEVICE="cuda:0"
EPOCHS=200

python tasks/multi_modal_task/molcap.py \
--device ${DEVICE} \
--config_path ./configs/molcap/${MODEL}-molt5-multnodes.json \
--dataset chebi-20 \
--dataset_path ../datasets/molcap/chebi-20 \
--output_path ../ckpts/finetune_ckpts/molcap-multnodes-attn/ \
--caption_save_path ../tmps/molcap/${MODEL}-captions.txt \
--mode ${MODE} \
--epochs ${EPOCHS} \
--num_workers 1 \
--batch_size 8 \
--logging_steps 300 \
--patience 200 \
--text2mol_bert_path ../ckpts/bert_ckpts/scibert_scivocab_uncased/ \
--text2mol_data_path ../assets/text2mol_data/ \
--text2mol_ckpt_path ../ckpts/fusion_ckpts/text2mol/test_outputfinal_weights.320.pt