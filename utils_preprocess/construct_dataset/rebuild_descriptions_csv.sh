#!/bin/bash

DATA_ROOT="/data/ironman/jiacheng/final_Omni_Data/test/full_0.05"
MAPPING_JSON="${DATA_ROOT}/mapping.json"
OUTPUT_CSV="/home/jiacheng/Omni_detection/PIXAR/utils_preprocess/descriptions_rebuilt.csv"

python3 /home/jiacheng/Omni_detection/PIXAR/utils_preprocess/construct_dataset/rebuild_descriptions_csv.py \
    --mapping-json "${MAPPING_JSON}" \
    --data-root "${DATA_ROOT}" \
    --output-csv "${OUTPUT_CSV}" \
    --splits validation
