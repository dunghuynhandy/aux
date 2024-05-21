# dataset settings
_base_ = 'coco_instance.py'
dataset_type = 'LVISV1Dataset'
data_root = 'data/lvis_v1/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            # ann_file=data_root + 'annotations/lvis_v1_val.json',
            ann_file=data_root + 'annotations/instances_val2017.json',
            img_prefix=data_root)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        # ann_file=data_root + 'annotations/lvis_v1_val.json',
        img_prefix="/mnt/datasets/llava_data/llava_second_stage/"),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/lvis_v1_val.json',
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix='/home/ngoc/githubs/aux/object_detection/Co-DETR/data/text_vqa/test_images'))
evaluation = dict(metric=['bbox', 'segm'])
