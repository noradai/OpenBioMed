#!/bin/bash
DATA_PATH="/share/project/molpretrain/data/"
INPUT_FILE=${DATA_PATH}"all_smiles.txt"
FP_CLU_FILE=${DATA_PATH}"fp.clu"
OUTPUT_FILE=${DATA_PATH}"output.txt"
THRESHOLD=0.95

if [ ! -f ${FP_CLU_FILE} ];
then
    echo "Generating Molecular Fingerprints"
    python feat/drug_featurizer.py \
    --mode file \
    --featurizer fp \
    --config_file configs/fp_featurizer.json \
    --smiles_file ${INPUT_FILE} \
    --output_file ${FP_CLU_FILE} \
    --post_transform to_clu
fi

cd ../assets/l2ap
make
cd build
./apss -t ${THRESHOLD} -sim tan mmj ${FP_CLU_FILE} ${OUTPUT_FILE} 