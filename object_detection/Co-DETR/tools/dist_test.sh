CONFIG=/home/ngoc/githubs/aux/object_detection/Co-DETR/projects/configs/co_dino/co_dino_5scale_lsj_vit_large_lvis.py
CHECKPOINT=/home/ngoc/githubs/aux/object_detection/Co-DETR/co_dino_5scale_lsj_vit_large_lvis.pth
GPUS=4
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --format-only --options "jsonfile_prefix=./results"