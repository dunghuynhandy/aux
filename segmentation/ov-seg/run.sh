#!/bin/bash
JSON_PATH=/home/ngoc/githubs/aux/object_detection/image_samples.json
IMG_PATH=/mnt/datasets/llava_data/llava_second_stage/
DETECTION_OUTPUT_PATH="/home/ngoc/githubs/aux/results/final_result_DET.json"
python ovseg_DDP.py --num-gpu 4 --config-file configs/ovseg_swinB_vitL_demo.yaml