#!/bin/bash

DIR_A="/data/ironman/jiacheng/final_Omni_Data/train/ours_0.05/train/metadata"
DIR_B="/data/thor/jiacheng/omni_backup/train/ours_0.1/train/metadata"

python3 /home/jiacheng/Omni_detection/PIXAR/utils_preprocess/construct_dataset/verify_metadata.py \
    "${DIR_A}" \
    "${DIR_B}"
