#!/bin/bash
hf_dataset='ai2d'

NNODES=1
NPROC_PER_NODE=4
MASTER_ADDR='localhost'
MASTER_PORT=7778
NODE_RANK=0

# JSON_PATH=/home/ngoc/githubs/aux/OCR/PaddleOCR/debuging.json
cache_dir="/home/ngoc/githubs/aux/custom_datasets"
mkdir -p ./ocr_output
mkdir -p ./results/$hf_dataset
CUDA_VISIBLE_DEVICES=0,1,2,3 WORLD_SIZE=$NPROC_PER_NODE python3 -m torch.distributed.launch \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --nnodes=$NNODES \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --use_env ocr.py \
    --hf_dataset $hf_dataset \
    --cache_dir $cache_dir \
    --bz_per_gpu 32