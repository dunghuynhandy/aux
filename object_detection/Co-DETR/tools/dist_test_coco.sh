CONFIG=/home/ngoc/githubs/aux/object_detection/Co-DETR/projects/configs/co_dino/co_dino_5scale_swin_large_16e_o365tococo.py
CHECKPOINT=/home/ngoc/githubs/aux/object_detection/Co-DETR/co_dino_5scale_swin_large_16e_o365tococo.pth
GPUS=4
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox