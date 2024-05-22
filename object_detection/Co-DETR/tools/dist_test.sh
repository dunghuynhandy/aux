#!/bin/bash
mkdir det_output
mkdir log_output
CONFIG=/home/ngoc/githubs/aux/object_detection/Co-DETR/projects/configs/co_dino/co_dino_5scale_lsj_vit_large_lvis.py
CHECKPOINT=/home/ngoc/githubs/aux/object_detection/Co-DETR/co_dino_5scale_lsj_vit_large_lvis.pth
GPUS=4
PORT=${PORT:-29500}

HF_DATASETS=(
    'scienceqa' 'aokvqa' 'chart2text' 'chartqa' 'clevr' 'clevr_math' 'cocoqa' 'datikz'
    'diagram_image_to_text' 'docvqa' 'dvqa' 'figureqa' 'finqa' 'geomverse' 'hateful_memes'
    'hitab' 'iam' 'iconqa' 'infographic_vqa' 'intergps' 'localized_narratives' 'mapqa'
    'mimic_cgd' 'multihiertt' 'nlvr2' 'ocrvqa' 'okvqa' 'plotqa' 'raven' 'rendered_text'
    'robut_sqa' 'robut_wikisql' 'robut_wtq' 'screen2words' 'spot_the_diff'
    'st_vqa' 'tabmwp' 'tallyqa' 'tat_qa' 'textcaps' 'textvqa' 'tqa' 'vistext' 'visual7w'
    'visualmrc' 'vqarad' 'vqav2' 'vsr' 'websight'
)

HF_DATASETS=(
    'scienceqa'
)

for DATASET in "${HF_DATASETS[@]}"; do
    echo "*********************************************************************************************************************************************************"
    echo "Running for dataset: $DATASET"
    CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch --format-only --options "jsonfile_prefix=./det_output/results_" --hf-dataset $DATASET --cache-dir '~/.cache/huggingface/datasets/' \
    # &> "./log_output/log_${DATASET}.txt"
done