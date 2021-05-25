import os
import gc
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import utils
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import Normalize

from transforms import ToTensor, Resize, Compose, RandomHorizontalFlip, RandomVerticalFlip, Rotation
from data.coco import Coco
from data.kitti_dataset import KITTI
from data.voc import VOC
from backbone.backbone_vgg import vgg16, vgg11, vgg13, vgg19
from backbone.backbone_resnet import resnet18, resnet34
from models.faster_rcnn_mod import FasterRCNN, fasterrcnn_resnet50_fpn
from models.mask_rcnn_mod import MaskRCNN
from models.keypoint_rcnn_mod import KeypointRCNN
from models.retinanet_mod import RetinaNet
from engine import train_one_epoch, evaluate
from matplotlib.patches import Rectangle

def parse_args():
    
    parser = argparse.ArgumentParser(description='Analyze Intrinsic dimension')
    parser.add_argument('--dataset', dest='dataset', help='training dataset', default='pascal_voc', type=str)
    parser.add_argument('--backbone', dest='backbone', help='vgg16, resnet_18', default='vgg16', type=str)
    parser.add_argument('--model', dest='model', help='faster_rcnn, retinanet', default='faster_rcnn', type=str)
    args = parser.parse_args()
    return args

args = parse_args()
root = os.getcwd()
root = os.path.split(root)[0]
model_path = os.path.join(root, "trained_model", args.dataset, args.model,args.backbone)

def collate_fn(batch):
    return tuple(zip(*batch))


def transform(width, height):
    transform = []
    transform.append(Resize((height, width)))
    transform.append(ToTensor())
    #transform.append(RandomHorizontalFlip(1))
    transform.append(RandomVerticalFlip(1))
    #transform.append(Rotation(1))
    transform = Compose(transform)
    return transform

if args.dataset == 'pascal_voc':
    root = '/work/chawda/Dataset/VOCdevkit/VOC2012'
    num_classes = 20
    width, height = 300, 300
    transform = transform(width, height)
    data = VOC(root, transform)
    indices = torch.randperm(len(data)).tolist()
    #dataset = torch.utils.data.Subset(data, indices[:13500])
    testdata = torch.utils.data.Subset(data, indices[13500:])
    #traindata = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    data_loader_test = DataLoader(testdata, batch_size=8, shuffle=True, collate_fn=collate_fn)
elif args.dataset == 'kitti':
    root = '/work/chawda/Dataset/KITTI'
    num_classes = 10
    width, height = 300, 300
    transform = transform(width, height)
    data = KITTI(root,transform)
    indices = torch.randperm(len(data)).tolist()
    #dataset = torch.utils.data.Subset(data, indices[:5500])
    testdata = torch.utils.data.Subset(data, indices[5500:])
    #traindata = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    data_loader_test = DataLoader(testdata, batch_size=4, shuffle=True, collate_fn=collate_fn)
else:
    root = '/work/chawda/Dataset/COCO/train2017/'
    annotations = '/work/chawda/Dataset/COCO/annotations/instances_train2017.json'
    width, height = 300, 300
    transform = transform(width, height)
    num_classes=91
    data = Coco(root = root, annotations=annotations, transforms=transform)
    indices = torch.randperm(len(data)).tolist()
    #dataset = torch.utils.data.Subset(data, indices[:80000])
    testdata = torch.utils.data.Subset(data, indices[80000:])
    #traindata = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    data_loader_test = DataLoader(testdata, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
backbone_name = args.backbone
if backbone_name == 'vgg16':
    backbone = vgg16(pretrained=True).features
    backbone.out_channels = 512
    res_anchor = 0 
elif backbone_name == 'vgg19':
    backbone = vgg19(pretrained=True).features
    backbone.out_channels = 512
    res_anchor = 0
elif backbone_name == 'resnet_50':
    resnet_net = torchvision.models.resnet50(pretrained=True)
    modules = list(resnet_net.children())[:-1]
    backbone = nn.Sequential(*modules)
    backbone.out_channels = 2048
    res_anchor = 1
elif backbone_name == 'resnet_101':
    resnet_net = torchvision.models.resnet101(pretrained=True)
    modules = list(resnet_net.children())[:-1]
    backbone = nn.Sequential(*modules)
    backbone.out_channels = 2048
    res_anchor = 1


if res_anchor == 1:
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0','1'], output_size=7, sampling_ratio=2)
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5,1.0,2.0),))
else:
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5,1.0,2.0),))

if args.model == 'faster_rcnn':
    model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator= anchor_generator, box_roi_pool=roi_pooler)
elif args.model == 'retinanet':
    model = RetinaNet(backbone, num_classes=num_classes, anchor_generator=anchor_generator)

if(torch.cuda.is_available()):
    device = torch.device("cuda")
    print(device, torch.cuda.get_device_name(0))
    print(torch.cuda.device_count())
else:
    device = torch.device("cpu")
    print(device)

if args.dataset == 'coco' and args.backbone == 'vgg16':
	model.load_state_dict(torch.load(os.path.join(model_path, 'checkpoint_20.pt'))['model_state_dict'])
else:
	model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt')))

model.to(device)
model.eval()
evaluate(model, data_loader_test, device=device)
