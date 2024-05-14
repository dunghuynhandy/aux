
import argparse
import json
import os
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.core import DatasetEnum
from PIL import Image
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
            
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
    

class OCRDataset(Dataset):
    def __init__(self, json_file, image_path):
        """
        Args:
            json_file (string): Path to the json file.
            img_dir (string): Directory with all the images.
            processor: Directory
        """
        
        with open(json_file, 'r') as file:
            self.images = json.load(file)
        
        
        with open("/home/ngoc/githubs/aux/results/json_files/final_result_CoDetr.json", 'r') as file:
            self.processed_images = json.load(file)
        self.processed_images = [item["img"] for item in self.processed_images]
        
        self.images = [item["image"] for item in self.images]
        self.images = list(set(self.images))
        self.images = [item for item in self.images if item not in self.processed_images]
        print(f"there are {len(self.images)} images")
        self.image_path = image_path
        
                
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item_file = self.images[idx]
        image_path = os.path.join(self.image_path, item_file)
        return {
            "image_path":image_path,
            }

def custom_collate_fn(batch):
    batch_image_paths = [item['image_path'] for item in batch]
    
    return {
        'image_paths': batch_image_paths,
    }
    
def collect_result(world_size):
    rew_result = []
    result = []
    
    for rank in range(world_size):
        with open(f"./results/data_rank_{rank}.jsonl", 'r') as file:
            for line in file:
                json_object = json.loads(line)
                result.append(json_object)
    print(f"the total of predictions: {len(result)}")
    unique_id = []
    for item in result:
        if item["img"] not in unique_id:
            rew_result.append(item)
            unique_id.append(item["img"])
    print(f"the total of unique predictions: {len(rew_result)}")

    return rew_result

def main(args):
    
    init_distributed_mode(args)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    OCR_dataset = OCRDataset(args.json_file, args.image_path)
    sampler = torch.utils.data.DistributedSampler(OCR_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    lvis_label_file = '/home/ngoc/githubs/aux/object_detection/labels_setup/ivis_labels.txt'
    lvis_label_mapping = {}
    idx = 0
    with open(lvis_label_file, 'r') as file:
        for line in file:
            lvis_label_mapping[idx] = line.strip()
            idx += 1
    
    config_file = '/home/ngoc/githubs/aux/object_detection/Co-DETR/projects/configs/co_dino/co_dino_5scale_lsj_vit_large_lvis.py'
    checkpoint_file = '/home/ngoc/githubs/aux/object_detection/Co-DETR/co_dino_5scale_lsj_vit_large_lvis.pth'

    model = init_detector(config_file, checkpoint_file, DatasetEnum.LVIS, device=args.device)
    dataset_loader = DataLoader(OCR_dataset,
                                sampler=sampler,
                                batch_size=args.bz_per_gpu, 
                                num_workers=4,
                                shuffle = False,
                                collate_fn=custom_collate_fn)
    
    for idx, batch_data in enumerate(dataset_loader):
        if rank == 0:
            print(f"[{idx}|{len(dataset_loader)}]")
        for image_pth in batch_data["image_paths"]:
            bbox_result = inference_detector(model, image_pth)
            labels = [np.full(bbox.shape[0], i, dtype=np.int32)\
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)
            bboxes = np.vstack(bbox_result)
            labels_impt = np.where(bboxes[:, -1] > args.threshold)[0]
            if len(labels_impt) > 0:
                labels_impt_list = [labels[i] for i in labels_impt]
                labels_class = [lvis_label_mapping[i] for i in labels_impt_list]
                filter_bbox = bboxes[labels_impt]
                
                bounding_boxes = filter_bbox[:, :4].tolist()
                scores = filter_bbox[:, 4].tolist()
                
            else:
                bounding_boxes = []
                scores = []
                labels_class = []
            item = {
                    "img": image_pth.replace(args.image_path, ''),
                    "texts": labels_class,
                    "bboxes": bounding_boxes,
                    "scores": scores
                }
            with open(f'./results/data_rank_{rank}.jsonl', 'a') as file:
                    json.dump(item, file)
                    file.write('\n')
    dist.barrier()
    if rank == 0:
        final_result = collect_result(4)
        output_path = os.path.join(args.output_path, "final_result_CoDetr.json") 
        with open(output_path, 'w') as file:
            json.dump(final_result, file, indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--json_file', default='/home/ngoc/githubs/aux/image_sample.json')
    parser.add_argument('--image_path',  default='/home/ngoc/data/')
    parser.add_argument('--bz_per_gpu', default=4, type=int)
    parser.add_argument('--output_path', default='/home/ngoc/githubs/aux/results/json_files')
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    main(args)