from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pickle
import argparse
from tqdm import tqdm
from torchvision.models.detection.rpn import AnchorGenerator
import os


from intrinsic_dimension import estimate, block_analysis
from scipy.spatial.distance import pdist,squareform

from backbone.backbone_vgg import vgg16
data_folder = ('Dataset/KITTI/testing/')
data_transform = transforms.Compose([
        transforms.Resize( (224,224) , interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

image_dataset = datasets.ImageFolder(data_folder, data_transform)
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=20, 
                                         shuffle=True, num_workers=1)

use_gpu = torch.cuda.is_available()
from models.faster_rcnn_mod import FasterRCNN, TwoMLPHead, FastRCNNPredictor

backbone = vgg16(pretrained=False).features
backbone.out_channels = 512

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256),), aspect_ratios=((0.5,1.0,2.0),)) 

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

class WrappedModel(nn.Module):
	def __init__(self):
		super(WrappedModel, self).__init__()
		self.module = FasterRCNN(backbone, num_classes=10,rpn_anchor_generator= anchor_generator, box_roi_pool=roi_pooler ) # that I actually define.
	def forward(self, x):
		return self.module(x)

model = WrappedModel()
model = nn.DataParallel(model)
model.load_state_dict(torch.load('model1.pt'))

model.eval()
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print(model)
