#!/bin/bash

CONFIG=./projects/configs/co_dino/co_dino_5scale_lsj_vit_large_lvis.py
CHECKPOINT=./co_dino_5scale_lsj_vit_large_lvis.pth
GPUS=4
PORT=${PORT:-29500}
custom_dataset_save_path=../../custom_datasets
cache_dir='~/.cache/huggingface/datasets'
hf_datasets=('ai2d' 'aokvqa' 'chart2text' 'chartqa' 'clevr' 'clevr_math' 
'cocoqa' 'datikz' 'diagram_image_to_text' 'docvqa' 'dvqa' 'figureqa' 
'finqa' 'geomverse' 'hateful_memes' 'hitab' 'iam' 'iconqa' 'infographic_vqa' 
'intergps' 'localized_narratives' 'mapqa' 'mimic_cgd' 'multihiertt' 'nlvr2' 
'ocrvqa' 'okvqa' 'plotqa' 'raven' 'rendered_text' 'robut_sqa' 'robut_wikisql' 
'robut_wtq' 'scienceqa' 'screen2words' 'spot_the_diff' 'st_vqa' 'tabmwp' 'tallyqa' 
'tat_qa' 'textcaps' 'textvqa' 'tqa' 'vistext' 'visual7w' 'visualmrc'
'vqarad' 'vqav2' 'vsr' 'websight')
hf_datasets=('aokvqa' 'ocrvqa' 'okvqa'  'textcaps' 'vqav2')
hf_datasets=('okvqa')
for hf_dataset in "${hf_datasets[@]}"; do
  echo '*********************************************************************************************************************************************************************************'
  echo '*********************************************************************************************************************************************************************************'
  echo '*********************************************************************************************************************************************************************************'
  echo '*********************************************************************************************************************************************************************************'
  echo $hf_dataset
  mkdir -p ${custom_dataset_save_path}/${hf_dataset}
  mkdir -p logs
  log_file="logs/${hf_dataset}.txt"
  CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
  python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --format-only --options "jsonfile_prefix=./det_output/results_" --hf-dataset $hf_dataset --cache-dir $cache_dir --custom-dataset-save-path $custom_dataset_save_path \
    2>&1 | tee $log_file
done