from paddleocr import PaddleOCR
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
from PIL import Image
from torch.utils.data.distributed import DistributedSampler
from datasets import load_from_disk
import tqdm
import cv2
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                        world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()

def pil_to_numpy(example):
    example['image'] = example['images'][0]
    
    del example['images']
    return example

class OCRDataset(Dataset):
    def __init__(self, cache_dir, hf_dataset):
        """
        Args:
            json_file (string): Path to the json file.
            img_dir (string): Directory with all the images.
            processor: Directory
        """
        hf_folder = os.path.join(cache_dir,hf_dataset)
        self.images = load_from_disk(hf_folder)["train"]
        # self.images = self.images.map(pil_to_numpy)
        print(f"there are {len(self.images)} images")
        
                
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        return {
            "image":image["images"][0],
            "image_id": image["id"]
            }

def custom_collate_fn(batch):
    batch_images = [item['image'] for item in batch]
    batch_ids = [item['image_id'] for item in batch]
    
    return {
        'images': batch_images,
        "ids": batch_ids
    }
import subprocess
def kill_train_processes():
    # Command to find and kill all processes running train.py
    command = "kill $(ps aux | grep 'run.sh' | grep -v grep | awk '{print $2}')"
    
    try:
        # Run the command
        subprocess.run(command, shell=True, check=True)
        print("Processes running train.py have been killed.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
def cleanup():
    # Clean up and destroy the process group
    if dist.is_initialized():
        dist.destroy_process_group()
def collect_result(world_size, hf_dataset):
    rew_result = {}
    result = []
    
    for rank in range(world_size):
        with open(f"./results/{hf_dataset}/data_rank_{rank}.jsonl", 'r') as file:
            for line in file:
                json_object = json.loads(line)
                result.append(json_object)
    
    print(f"the total of predictions: {len(result)}")
    for item in result:
        if item["id"] not in rew_result.keys():
            img_id = item["id"]
            del item["id"]
            rew_result[img_id] = item
    print(f"the total of unique predictions: {len(rew_result)}")

    return rew_result

def convert_to_xyminmax(bboxes):
    converted_bboxes = []
    for corners in bboxes:
        x_coords = [point[0] for point in corners]
        y_coords = [point[1] for point in corners]
        xmin = min(x_coords)
        ymin = min(y_coords)
        xmax = max(x_coords)
        ymax = max(y_coords)
        converted_bboxes.append([xmin, ymin, xmax, ymax])
    
    return converted_bboxes
def main(args):
    init_distributed_mode(args)
    rank = dist.get_rank()
    device = torch.device(args.device)
    world_size = get_world_size()
    
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    OCR_dataset = OCRDataset(args.cache_dir, args.hf_dataset)

    samplers = DistributedSampler(OCR_dataset, rank=rank, num_replicas=world_size)
    test_kwargs = {'batch_size': args.bz_per_gpu, 'sampler': samplers, 'shuffle': False, 'pin_memory' :True, 'num_workers':4}
    
    dataset_loader = DataLoader(OCR_dataset,
                                collate_fn=custom_collate_fn,
                                **test_kwargs)
    ocr = PaddleOCR(use_angle_cls = True,lang="en", gpu_id = rank, use_gpu=True, warmup=False, benchmark=True, show_log = False)
    for idx, batch_data in enumerate(dataset_loader):
        if rank == 0:
            print(f"[{idx}|{len(dataset_loader)}]")
        for img_id, img in zip(batch_data["ids"], batch_data["images"]):
            image_np = np.array(img)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            result = ocr.ocr(image_np, cls=False)[0]
            if result:
                boxes = [line[0] for line in result]
                boxes = convert_to_xyminmax(boxes)
                txts = [line[1][0] for line in result]
                scores = [line[1][1] for line in result]
            else:
                boxes = []
                txts = []
                scores = []

            item = {
                "id": img_id,
                "texts": txts,
                "bboxes": boxes,
                "scores": scores
                
            }
            with open(f'./results/{args.hf_dataset}/data_rank_{rank}.jsonl', 'a') as file:
                    json.dump(item, file)
                    file.write('\n')
    dist.barrier()
    if rank == 0:
        print("combining results")
        final_result = collect_result(world_size, args.hf_dataset)
        output_path = os.path.join("./ocr_output/",f"results_{args.hf_dataset}.ocr.json") 
        with open(output_path, 'w') as file:
            json.dump(final_result, file)
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--hf_dataset', default='ai2d')
    parser.add_argument('--cache_dir',  default='/home/ngoc/githubs/aux/custom_datasets')
    parser.add_argument('--bz_per_gpu', default=4, type=int)
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--output_path', default='/home/ngoc/githubs/aux/results/json_files')
    args = parser.parse_args()
    
    main(args)
    quit()