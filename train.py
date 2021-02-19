#https://github.com/jwyang/faster-rcnn.pytorch/blob/master/trainval_net.py
import os
import gc
import sys
import argparse
import time
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import utils
from torchvision.models.detection.rpn import AnchorGenerator
from utils.transforms import Compose, ToTensor, RandomHorizontalFlip
from data.coco import Coco
from data.kitti_dataset import KITTI
from data.voc import VOC
from backbone.backbone_vgg import vgg16
from models.faster_rcnn_mod import FasterRCNN
from utils.engine import train_one_epoch, evaluate

def parse_args():
    
    parser = argparse.ArgumentParser(description='Analyze Intrinsic dimension')
    parser.add_argument('--dataset', dest='dataset', help='training dataset', default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net', help='vgg16, res101', default='vgg16', type=str)
    parser.add_argument('--bs', dest='batch_size', help='batch_size', default=1, type=int)

    args = parser.parse_args()
    return args

args = parse_args()

transform = []
transform.append(ToTensor())
transform = Compose(transform)

if args.dataset == 'pascal_voc':
    root = '/home/ajay/Desktop/ID_CNN/Dataset/VOCdevkit/VOC2012'
    trainset = VOC(root, transform)
elif args.dataset == 'kitti':
    root = '/home/ajay/Desktop/ID_CNN/Dataset/KITTI'
    trainset = KITTI(root,transform)
else:
    root = '/home/ajay/Desktop/ID_CNN/Dataset/COCO/train2017/'
    annotations = '/home/ajay/Desktop/ID_CNN/Dataset/COCO/annotations/instances_train2017.json'
    trainset = Coco(root = root, annotations=annotations)
    
def collate_fn(batch):
    return tuple(zip(*batch))

print(len(trainset))
indices = torch.randperm(len(trainset)).tolist()
dataset = torch.utils.data.Subset(trainset, indices[:1000])
testdata = torch.utils.data.Subset(trainset, indices[7000:])
traindata = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
data_loader_test = DataLoader(testdata, batch_size=4, shuffle=True, collate_fn=collate_fn)



if(torch.cuda.is_available()):
    device = torch.device("cuda")
    print(device, torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print(device)




backbone = vgg16(pretrained=True).features
backbone.out_channels = 512

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256),), aspect_ratios=((0.5,1.0,2.0),)) 

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

model = FasterRCNN(backbone, num_classes=10, rpn_anchor_generator= anchor_generator, box_roi_pool=roi_pooler)

model.to(device)

#model = nn.DataParallel(model)


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 10

for epoch in range(num_epochs):
    
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, traindata, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)
   
    
        
torch.save(model.state_dict(), 'model1.pt')



