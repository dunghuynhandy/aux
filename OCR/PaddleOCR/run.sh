#!/bin/bash
NNODES=1
NPROC_PER_NODE=4
MASTER_ADDR='localhost'
MASTER_PORT=7778
NODE_RANK=0

# JSON_PATH=/home/ngoc/githubs/aux/OCR/PaddleOCR/debuging.json
JSON_PATH=/home/ngoc/githubs/aux/object_detection/Co-DETR/data/text_vqa/TextVQA_0.5.1_test.json
IMG_PATH=/home/ngoc/githubs/aux/object_detection/Co-DETR/data/text_vqa/test_images

CUDA_VISIBLE_DEVICES=0,1,2,3  WORLD_SIZE=$NPROC_PER_NODE python3 -m torch.distributed.launch \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --nnodes=$NNODES \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --use_env ocr.py \
    --json_file $JSON_PATH \
    --image_path $IMG_PATH \
    --bz_per_gpu 4