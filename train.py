#https://github.com/jwyang/faster-rcnn.pytorch/blob/master/trainval_net.py
import os
import sys
import argparse
import time

import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from torchvision import utils, transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from data.coco import Coco
from data.kitti_dataset import KITTI
from data.voc import VOC

def parse_args():
    
    parser = argparse.ArgumentParser(description='Analyze Intrinsic dimension')
    parser.add_argument('--dataset', dest='dataset', help='training dataset', default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net', help='vgg16, res101', default='vgg16', type=str)
    parser.add_argument('--bs', dest='batch_size', help='batch_size', default=1, type=int)

    args = parser.parse_args()
    return args

args = parse_args()

def get_transform(train):
    transform=[]
    transform.append(transforms.ToTensor())
    if train:
        transform.append(transforms.Normalize(0.485, 0.456, 0.406))
    return transforms.Compose(transform)

if args.dataset == 'pascal_voc':
    root = 'D:/Dataset/VOCdevkit/VOC2012'
    trainset = VOC(root, None)
elif args.dataset == 'kitti':
    root = 'D:/Dataset/KITTI'
    trainset = KITTI(root,None)
else:
    root = 'D:/Dataset/COCO/train2017/'
    annotations = 'D:/Dataset/COCO/annotations/instances_train2017.json'
    trainset = Coco(root = root, annotations=annotations)
    
def collate_fn(batch):
    return tuple(zip(*batch))

traindata = DataLoader(trainset, batch_size=4, shuffle=True, collate_fn=collate_fn)

print(trainset)
print(traindata)

images, target = next(iter(traindata))

print(target)
if(torch.cuda.is_available()):
    device = torch.device("cuda")
    print(device, torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print(device)



