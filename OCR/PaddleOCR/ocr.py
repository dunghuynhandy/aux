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
        
        self.images = self.images["data"]
        # import ipdb; ipdb.set_trace()
        self.images = [item["image_id"]+".jpg" for item in self.images]
        # self.images = [item["image"] for item in self.images if "image" in item]
        
        self.images = list(dict.fromkeys(self.images))

        
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

    OCR_dataset = OCRDataset(args.json_file, args.image_path)

    samplers = DistributedSampler(OCR_dataset, rank=rank, num_replicas=world_size)
    test_kwargs = {'batch_size': args.bz_per_gpu, 'sampler': samplers, 'shuffle': False, 'pin_memory' :True, 'num_workers':4}
    
    dataset_loader = DataLoader(OCR_dataset,
                                collate_fn=custom_collate_fn,
                                **test_kwargs)
    ocr = PaddleOCR(lang='en',use_gpu = True, use_mp = True, show_log = False)
    for idx, batch_data in enumerate(dataset_loader):
        if rank == 0:
            print(f"[{idx}|{len(dataset_loader)}]")
        for img_path in batch_data["image_paths"]:
            result = ocr.ocr(img_path, cls=False)[0]
            if len(result) > 0:
                boxes = [line[0] for line in result]
                boxes = convert_to_xyminmax(boxes)
                txts = [line[1][0] for line in result]
                scores = [line[1][1] for line in result]
            else:
                boxes = []
                txts = []
                scores = []

            item = {
                "img": img_path.replace('/mnt/datasets/llava_data/llava_second_stage/', ''),
                "texts": txts,
                "bboxes": boxes,
                "scores": scores
                
            }
            with open(f'./results/data_rank_{rank}.jsonl', 'a') as file:
                    json.dump(item, file)
                    file.write('\n')
    dist.barrier()
    # if rank == 0:
    #     final_result = collect_result(4)
    #     output_path = os.path.join(args.output_path, "final_result_PaddleOCR_first_stage.json") 
    #     with open(output_path, 'w') as file:
    #         json.dump(final_result, file, indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--json_file', default='/home/ngoc/githubs/aux/image_sample.json')
    parser.add_argument('--image_path',  default='/home/ngoc/data/')
    parser.add_argument('--bz_per_gpu', default=4, type=int)
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--output_path', default='/home/ngoc/githubs/aux/results/json_files')
    args = parser.parse_args()
    
    main(args)