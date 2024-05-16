import argparse
import json
import math
import os
import random
import shutil
import string
import time
import warnings
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from PIL import Image
import multiprocessing
from functools import partial
import pickle
import easyocr

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
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
        # self.images = [item["image"] for item in self.images if ('image' in item) and 'coco' in item['image']]
        self.images = list(set(self.images))
        print(f"there are {len(self.images)} images")
        self.image_path = image_path
        
                
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item_file = self.images[idx]
        image_path = os.path.join(self.image_path, item_file)
        # image = Image.open(image_path)
        # image = image.convert('RGB')
        return {
            "image_path":image_path,
            # "image":image
            }

def custom_collate_fn(batch):
    batch_image_paths = [item['image_path'] for item in batch]
    # images = [item['image'] for item in batch]
    
    return {
        'image_paths': batch_image_paths,
        # 'images': images
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
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    OCR_dataset = OCRDataset(args.json_file, args.image_path)
    sampler = torch.utils.data.DistributedSampler(OCR_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    
    dataset_loader = DataLoader(OCR_dataset,
                                sampler=sampler,
                                batch_size=args.bz_per_gpu, 
                                num_workers=4,
                                shuffle = False,
                                collate_fn=custom_collate_fn)
    ocr = easyocr.Reader(['ch_sim','en'])
    for idx, batch_data in enumerate(dataset_loader):
        print(f"[{idx}|{len(dataset_loader)}]")
        for img_path in batch_data["image_paths"]:
            result = ocr.readtext(img_path)
            if len(result) > 0:
                boxes = [line[0] for line in result]
                boxes = json.dumps(boxes, cls=NumpyEncoder)
                txts = [line[1] for line in result]
                scores = [line[2] for line in result]
            else:
                boxes = []
                txts = []
                scores = []

            item = {
                "img": img_path.replace('/mnt/datasets/llava_data/llava_second_stage/', ''),
                "texts": txts,
                "boxes":boxes,
                "scores": scores
                
            }
            with open(f'./results/data_rank_{rank}.jsonl', 'a') as file:
                    json.dump(item, file)
                    file.write('\n')
    dist.barrier()
    if rank == 0:
        final_result = collect_result(world_size)
        output_path = os.path.join(args.output_path, "final_result_easyOCR.json") 
        with open(output_path, 'w') as file:
            json.dump(final_result, file, indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--json_file', default='/home/ngoc/githubs/aux/image_sample.json')
    parser.add_argument('--image_path',  default='/home/ngoc/data/')
    parser.add_argument('--bz_per_gpu', default=4, type=int)
    parser.add_argument('--output_path', default='/home/ngoc/githubs/aux/results/')
    args = parser.parse_args()
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.tensor([1, 2, 3], device=device)
    main(args)