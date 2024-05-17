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
import yaml
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from detectron2.engine import launch
from demo_predictor import demo_predictor
import detectron2.utils.comm as comm
import torch
from detectron2.data.detection_utils import read_image
from demo_predictor import demo_predictor
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
import zlib
import io
import base64
import matplotlib.pyplot as plt
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
    def __init__(self, json_file, image_path, detection_output_path):
        """
        Args:
            json_file (string): Path to the json file.
            img_dir (string): Directory with all the images.
            processor: Directory
        """
        with open(json_file, 'r') as file:
            self.images = json.load(file)
        # self.images = [item for item in self.images if ('image' in item) and 'coco' in item['image']][:128]
        self.image_path = image_path
        
        with open(detection_output_path, 'r') as file:
            self.detection_labels = json.load(file)
        self.detection_labels = {item["img"]:item["labels"] for item in self.detection_labels}
        print(f"There are {len(self.detection_labels)} open-word labels")
        print(f"There are {len(self.images)} images")
                
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_file = self.images[idx]
        open_word_labels = self.detection_labels[image_file]
        image_path = os.path.join(self.image_path, image_file)
        # image = Image.open(image_path)
        
        return {
            "image_path":image_path,
            # "image":image,
            "open_word_labels": open_word_labels
            }


def custom_collate_fn(batch):
    batch_image_paths = [item['image_path'] for item in batch]
    open_word_labels = [item['open_word_labels'] for item in batch]
    # images = [item['image'] for item in batch]
    
    return {
        'image_paths': batch_image_paths,
        'texts': open_word_labels,
        # 'images': images
    }
    
def encode_binary_mask(mask):
    """Converts a binary mask into OID challenge encoding ascii text."""

    # check input mask --
    
    mask = np.squeeze(mask)
    

    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)


    # compress and base64 encoding --
    binary_str = zlib.compress(mask_to_encode.tobytes(), zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str.decode('utf-8') 
    
def main(args):
    
    # init_distributed_mode(args)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    args.json_file = '/home/ngoc/githubs/aux/object_detection/image_samples.json'
    args.image_path = '/mnt/datasets/llava_data/llava_second_stage/'
    args.detection_output_path = "/home/ngoc/githubs/aux/results/final_result_DET.json"
    args.output_path = "/home/ngoc/githubs/aux/results"
    args.test_batch_size = 4
    Seg_dataset = SegDataset(args.json_file, args.image_path, args.detection_output_path)
    sampler = torch.utils.data.DistributedSampler(Seg_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataset_loader = DataLoader(Seg_dataset,
                                sampler=sampler,
                                batch_size=args.test_batch_size, 
                                num_workers=4,
                                shuffle = False,
                                collate_fn=custom_collate_fn)
    PS_model = demo_predictor()
    
    for idx, batch in enumerate(dataset_loader):
        if rank == 0:
            print(f"[{idx}| {len(dataset_loader)}]")
        image_paths = batch['image_paths']
        texts = batch['texts']
        for image_path, text in zip(image_paths, texts):
            try:
                img = read_image(image_path, format="BGR")
                predictions, visualized_output = PS_model.run_on_image(img, text)
                labels, bboxes = predictions["labels"], predictions["bboxes"]

                # segmentation = predictions["sem_seg"].cpu().numpy()
                segmentation = predictions["masks"]
                # segmentation = (segmentation < 0.5).astype(int)
                # segmentation = (segmentation >= 0.5).astype(int)
                segmentation = [encode_binary_mask(segment) for segment in segmentation]
                
                output_img = visualized_output.get_image()
                image_file = image_path.replace("/mnt/datasets/llava_data/llava_second_stage/", "")
                plt.imshow(output_img)
                plt.title('Output Image')
                plt.axis('off')
                plt.savefig(os.path.join("/home/ngoc/githubs/aux/segmentation/ov-seg/results/seg_img", image_file.replace("/", "-")))
            
                    
                
                
            except:
                labels, bboxes, segmentation = [], [], None
                

            item = {
                "img" : image_file,
                "labels": labels,
                "bboxes": bboxes,
                "segmentation": segmentation,
                "shape": img.shape[:-1]
            }
            with open(f'./results/data_rank_{rank}.jsonl', 'a') as file:
                json.dump(item, file)
                file.write('\n')
    dist.barrier()
    if rank == 0:
        final_result = collect_result(world_size)
        output_path = os.path.join(args.output_path, "final_result_SEG.json") 
        with open(output_path, 'w') as file:
            json.dump(final_result, file, indent=4)

if __name__ == '__main__':
    
    args = default_argument_parser().parse_args()
    
    
    
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


