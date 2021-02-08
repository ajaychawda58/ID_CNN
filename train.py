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

from pycocotools.coco import COCO
from coco import CocoDetection
from kitti_dataset import KittiLoader, AnnotationTransform_kitti

def parse_args():
    
    parser = argparse.ArgumentParser(description='Analyze Intrinsic dimension')
    parser.add_argument('--dataset', dest='dataset', help='training dataset', default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net', help='vgg16, res101', default='vgg16', type=str)
    parser.add_argument('--bs', dest='batch_size', help='batch_size', default=1, type=int)

    args = parser.parse_args()
    return args

args = parse_args()
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

if args.dataset == 'pascal_voc':
    root = '/home/ajay/Desktop/ID_CNN/Dataset/VOCdata'
    trainset = dset.VOCDetection(root=root, year='2012', image_set='train', download=False, transform = transform)
elif args.dataset == 'kitti':
    root = '/home/ajay/Desktop/ID_CNN/Dataset/KITTI'
    trainset = KittiLoader(root=root, split='training', target_transform = AnnotationTransform_kitti())
else:
    root = '/home/ajay/Desktop/ID_CNN/Dataset/COCO/train2017/'
    annotations = '/home/ajay/Desktop/ID_CNN/Dataset/COCO/annotations_COCO/annotations/instances_train2017.json'
    trainset = CocoDetection(root = root, annotations=annotations, transform=transform)
    
def collate_fn(batch):
    return tuple(zip(*batch))

traindata = DataLoader(trainset, batch_size=1, shuffle=True, collate_fn=collate_fn)

images, target = next(iter(traindata))

print(target)

if(torch.cuda.is_available()):
    device = torch.device("cuda")
    print(device, torch.cuda.get_device_name(0))
else:
    device(torch.device("cpu"))
    print(device)



