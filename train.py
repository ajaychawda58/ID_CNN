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

from transforms import ToTensor, Resize, Compose
from data.coco import Coco
from data.kitti_dataset import KITTI
from data.voc import VOC
from backbone.backbone_vgg import vgg16, vgg11, vgg13, vgg19
from backbone.backbone_resnet import resnet18, resnet34
from models.faster_rcnn_mod import FasterRCNN
from models.mask_rcnn_mod import MaskRCNN
from models.keypoint_rcnn_mod import KeypointRCNN
from models.retinanet_mod import RetinaNet
from engine import train_one_epoch, evaluate
from matplotlib.patches import Rectangle

def parse_args():
    
    parser = argparse.ArgumentParser(description='Analyze Intrinsic dimension')
    parser.add_argument('--dataset', dest='dataset', help='training dataset', default='pascal_voc', type=str)
    parser.add_argument('--backbone', dest='backbone', help='vgg16, resnet_18', default='vgg16', type=str)
    parser.add_argument('--bs', dest='batch_size', help='batch_size', default=1, type=int)
    parser.add_argument('--model', dest='model', help='faster_rcnn, retinanet', default='faster_rcnn', type=str)
    parser.add_argument('--epoch', dest='epoch', help='number of epochs', default=10, type=int)
    args = parser.parse_args()
    return args

args = parse_args()



save_path = os.path.join("trained_model", args.dataset, args.model,args.backbone, "model.pt")
checkpoint_path = os.path.join("trained_model", args.dataset, args.model,args.backbone)

def collate_fn(batch):
    return tuple(zip(*batch))

def draw_box(box, label, score=''):
    x1, y1, x2, y2 = box
    color = plt.cm.get_cmap('tab10')(label)
    rect = Rectangle((x1,y1),x2-x1,y2-y1,linewidth=2,
                     edgecolor='k',
                     facecolor=color,alpha=0.5)
    plt.gca().add_patch(rect)
    

def plot_img(image, target=None, thld=0.5):
    image = image.permute(1,2,0).numpy()
    plt.imshow(image)
    if target:
        for i in range(len(target['labels'])):
            scores = target.get('scores')
            if scores is not None:
                if scores[i]<thld:
                    continue
                draw_box(target['boxes'][i], int(target['labels'][i]), str(float(scores[i]))[:4])
            else:
                draw_box(target['boxes'][i], int(target['labels'][i]))

def transform(width, height):
    transform = []
    transform.append(Resize((height, width)))
    transform.append(ToTensor())
    transform = Compose(transform)
    return transform


if args.dataset == 'pascal_voc':
    root = '/work/chawda/Dataset/VOCdevkit/VOC2012'
    num_classes = 21
    width, height = 300, 300
    transform = transform(width, height)
    data = VOC(root, transform)
    indices = torch.randperm(len(data)).tolist()
    dataset = torch.utils.data.Subset(data, indices[:13500])
    testdata = torch.utils.data.Subset(data, indices[13500:])
    traindata = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    data_loader_test = DataLoader(testdata, batch_size=4, shuffle=True, collate_fn=collate_fn)
elif args.dataset == 'kitti':
    root = '/work/chawda/Dataset/KITTI'
    num_classes = 10
    width, height = 1200, 1200
    transform = transform(width, height)
    data = KITTI(root,transform)
    indices = torch.randperm(len(data)).tolist()
    dataset = torch.utils.data.Subset(data, indices[:5500])
    testdata = torch.utils.data.Subset(data, indices[5500:])
    traindata = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    data_loader_test = DataLoader(testdata, batch_size=4, shuffle=True, collate_fn=collate_fn)
else:
    root = '/work/chawda/Dataset/COCO/train2017/'
    annotations = '/work/chawda/Dataset/COCO/annotations/instances_train2017.json'
    width, height = 300, 300
    transform = transform(width, height)
    data = Coco(root = root, annotations=annotations, transforms=transform)
    indices = torch.randperm(len(data)).tolist()
    dataset = torch.utils.data.Subset(data, indices[:100000])
    testdata = torch.utils.data.Subset(data, indices[100000:])
    traindata = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    data_loader_test = DataLoader(testdata, batch_size=4, shuffle=True, collate_fn=collate_fn)
    num_classes = 91
    
print(len(data))


if(torch.cuda.is_available()):
    device = torch.device("cuda")
    print(device, torch.cuda.get_device_name(0))
    print(torch.cuda.device_count())
else:
    device = torch.device("cpu")
    print(device)

'''
img, trgt = dataset[2]
plt.figure(figsize=(20,20))
plot_img(img, trgt)
'''

backbone_name = args.backbone
if backbone_name == 'vgg16':
    backbone = vgg16(pretrained=True).features
    backbone.out_channels = 512
elif backbone_name == 'vgg19':
    backbone = vgg19(pretrained=True).features
    backbone.out_channels = 512
elif backbone_name == 'resnet_50':
    resnet_net = torchvision.models.resnet50(pretrained=True)
    modules = list(resnet_net.children())[:-1]
    backbone = nn.Sequential(*modules)
    backbone.out_channels = 2048
elif backbone_name == 'resnet_101':
    resnet_net = torchvision.models.resnet101(pretrained=True)
    modules = list(resnet_net.children())[:-1]
    backbone = nn.Sequential(*modules)
    backbone.out_channels = 2048


anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5,1.0,2.0),)) 

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

if args.model == 'faster_rcnn':
    model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator= anchor_generator, box_roi_pool=roi_pooler)
                       keypoint_roi_pool=keypoint_roi_pooler)
elif args.model == 'retinanet':
    model = RetinaNet(backbone, num_classes=num_classes, anchor_generator=anchor_generator)

model.to(device)


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

for epoch in range(num_epochs):
    	
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, traindata, device, epoch, print_freq=1000)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)
    print(epoch)
   
        
torch.save(model.state_dict(),save_path)



