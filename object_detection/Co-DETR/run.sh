#!/bin/bash
NNODES=1
NPROC_PER_NODE=4
MASTER_ADDR='localhost'
MASTER_PORT=7778
NODE_RANK=0
THRESHOLD=0.5

JSON_PATH=./LlavaFalcon_first_state_5k_images.json

# JSON_PATH=/mnt/datasets/llava_data/llava_second_stage/llava_v1_5_mix665k.json
IMG_PATH=/mnt/datasets/llava_data/llava_first_stage/images/

CUDA_VISIBLE_DEVICES=0,1,2,3 WORLD_SIZE=$NPROC_PER_NODE python3 -m torch.distributed.launch \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --nnodes=$NNODES \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --master_port=$MASTER_PORT \
    --use_env OD_detector.py \
    --json_file $JSON_PATH \
    --image_path $IMG_PATH \
    --bz_per_gpu 4 \
    --threshold $THRESHOLD