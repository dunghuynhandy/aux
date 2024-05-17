import os
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import pickle
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
import numpy as np
import random
from demo_predictor import demo_predictor
from detectron2.data.detection_utils import read_image
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
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

class SegDataset(Dataset):
    def __init__(self, json_file, image_path, detection_result):
        """
        Args:
            json_file (string): Path to the json file.
            img_dir (string): Directory with all the images.
            processor: Directory
        """
        
        with open(json_file, 'r') as file:
            self.images = json.load(file)[:32]
            
        # self.images = [item["image"] for item in self.images if ('image' in item)]
        # self.images = list(set(self.images))
        
        self.image_path = image_path
        self.open_word_labels = []
        
        with open(detection_result, 'r') as file:
            for line in file:
                self.open_word_labels.append(line.strip())
        
        print(f"There are {len(self.open_word_labels)} open-word labels")
        print(f"There are {len(self.images)} images")
                
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_file = self.images[idx]
        image_path = os.path.join(self.image_path, image_file)
        
        image = Image.open(image_path)
        image = image.convert('RGB')
        open_word_labels = self.open_word_labels
        return {
            "image_path":image_path,
            "image":image,
            "open_word_labels": open_word_labels
            }
        
def custom_collate_fn(batch):
    batch_image_paths = [item['image_path'] for item in batch]
    open_word_labels = [item['open_word_labels'] for item in batch]
    images = [item['image'] for item in batch]
    return {
        'image_paths': batch_image_paths,
        'texts': open_word_labels,
        'images': images
    }
        
def fsdp_main(rank, world_size, args):
    setup(rank, world_size)
    
    Seg_dataset = SegDataset(args.json_file, args.image_path, args.detection_output)
    print(Seg_dataset[0])
    Seg_samplers = DistributedSampler(Seg_dataset, rank=rank, num_replicas=world_size)
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': Seg_samplers}
    cuda_kwargs = {'num_workers': 2,
                    'pin_memory': True,
                    'shuffle': False}
    test_kwargs.update(cuda_kwargs)    
    test_loader = torch.utils.data.DataLoader(Seg_dataset, collate_fn=custom_collate_fn, **test_kwargs)
    
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=20000
    )
    
    torch.cuda.set_device(rank)
    
    Seg_model = demo_predictor()
    Seg_model = Seg_model.to(rank)
    
    Seg_model = FSDP(Seg_model, auto_wrap_policy=my_auto_wrap_policy)
    try:
        for idx, batch_data in enumerate(test_loader):
            if rank == 0:
                print(f"[{idx}|{len(test_loader)}]")
            image_paths = batch_data["image_paths"]
            images = batch_data["images"]
            texts = batch_data['texts']
            Seg_inputs = Seg_processor(text=texts, images=images, return_tensors="pt", truncation= True)
            Seg_inputs = {k: v.to("cuda") for k,v in Seg_inputs.items()}
            
        
            with torch.no_grad():
                Seg_outputs = Seg_model(**Seg_inputs)
            target_sizes = [(max(image.size), max(image.size))  for image in images]
            results = Seg_processor.post_process_object_detection(outputs=Seg_outputs, target_sizes=target_sizes, threshold=0.5)

            for i in range(len(results)):
                classes = texts[0]
                boxes, scores, labels = results[i]["boxes"].cpu().tolist(), results[i]["scores"].cpu().tolist(), results[i]["labels"].cpu().tolist()
                labels  = [classes[label] for label in labels]
                item = {
                "img": image_paths[i].replace("/mnt/datasets/llava_data/llava_second_stage/", ""),
                "boxes": boxes,
                "scores": scores,
                "labels": labels}

                with open(f'./results/data_rank_{rank}.jsonl', 'a') as file:
                    json.dump(item, file)
                    file.write('\n')
    except Exception as e:
        print("An error occurred:", e)
    dist.barrier()
    if rank == 0:
        final_result = collect_result(world_size)
        output_path = os.path.join(args.output, "final_result_DET.json") 
        with open(output_path, 'w') as file:
            json.dump(final_result, file, indent=4)
    cleanup()
    
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=8, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--json_file', type=str,
                        # default='/mnt/datasets/llava_data/llava_second_stage/llava_v1_5_mix665k.json')
                        default='./home/ngoc/githubs/aux/object_detection/image_samples.json')
    parser.add_argument('--image_path', type=str,
                        default='/mnt/datasets/llava_data/llava_second_stage/')
    parser.add_argument('--detection_output', type=str,
                        default='/home/ngoc/githubs/aux/results/final_result_DET.json')
    parser.add_argument('--output', type=str,
                        default='/home/ngoc/githubs/aux/results')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)