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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import utils, transforms, datasets
from torchvision.models.detection.rpn import AnchorGenerator
from torch.autograd import Variable

from transforms import ToTensor, Resize, Compose, testTensor
from data.kitti_dataset import KITTI
from backbone.backbone_vgg import vgg16, vgg11, vgg13, vgg19
from backbone.backbone_resnet import resnet18, resnet34
from models.faster_rcnn_mod import FasterRCNN
from models.retinanet_mod import RetinaNet
from collections import OrderedDict
from ID.intrinsic_dimension import estimate, block_analysis
from scipy.spatial.distance import pdist,squareform

from tqdm import tqdm
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Intrinsic dimension')
    parser.add_argument('--augment', dest='augment', help='testing dataset', default='test2017', type=str)
    parser.add_argument('--model', dest='test', help='test data model', default='kitti', type=str)
    args = parser.parse_args()
    return args
args = parse_args()	
augmentation = args.augment


#dataset class for loading test data
class test_data(object):
    def __init__(self, path, transforms):
        self.path = path
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(path))) 
       
    
    def __getitem__(self, idx):
        print(idx)
        img_path = os.path.join(self.path, self.imgs[idx])
        
        image = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
        return image
    
    def __len__(self):
        return len(self.imgs)

#path to saved model
root = os.getcwd()
root = os.path.split(root)[0]
if args.test=='pascal_voc':
	model_path = os.path.join(root, "trained_model", "pascal_voc", "faster_rcnn", "resnet_101")
	save_path = os.path.join("/scratch", "chawda", "trained_model", "pascal_voc", "faster_rcnn", "resnet_101")
	num_classes=20
	image_directory = os.path.join("/work/chawda/Dataset/VOCdevkit/VOC2007", str(augmentation))
	data = test_data(image_directory, transforms= testTensor())
	indices = torch.randperm(len(data))
	data = torch.utils.data.Subset(data, indices[:1000])
elif args.test == 'coco':
	model_path = os.path.join(root, "trained_model", "coco", "faster_rcnn", "resnet_101")
	save_path = os.path.join("/scratch","chawda", "trained_model", "coco", "faster_rcnn", "resnet_101")
	num_classes=91
	image_directory = os.path.join("/work/chawda/Dataset/COCO", str(augmentation))
	data = test_data(image_directory, transforms= testTensor())
	indices = torch.randperm(len(data))
	data = torch.utils.data.Subset(data, indices[:1000])
else:
	model_path = os.path.join(root, "trained_model", "kitti", "faster_rcnn", "resnet_101")
	save_path = os.path.join("/scratch", "chawda", "trained_model", "kitti", "faster_rcnn", "resnet_101")
	num_classes = 10
	image_directory = os.path.join("/work/chawda/Dataset/KITTI/testing", str(augmentation))
	data = test_data(image_directory, transforms= testTensor())
	indices = torch.randperm(len(data))
	data = torch.utils.data.Subset(data, indices[:400])

#feature_maps to load the saved model        
resnet_net = torchvision.models.resnet101(pretrained=True)
modules = list(resnet_net.children())[:-1]
backbone = nn.Sequential(*modules)
backbone.out_channels = 2048
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5,1.0,2.0),)) 
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1'], output_size=7, sampling_ratio=2)

#loading the model
model = FasterRCNN(backbone, num_classes=num_classes,rpn_anchor_generator= anchor_generator, box_roi_pool=roi_pooler )
model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt')))

fname = os.path.join(save_path, 'ID_all_'+str(augmentation))

model.cuda()
model.eval()


#weight extraction function
def extract_all(model, image, x, boxes, image_size):
	#backbone network
	out0 = image
	out1 = model.backbone[3](F.relu(model.backbone[1](model.backbone[0](out0))))
	out2 = model.backbone[4](out1)
	out3 = model.backbone[5](out2)
	out4 = model.backbone[6](out3)
	out5 = model.backbone[8](model.backbone[7](out4))
	#region proposal network
	out6 = model.rpn.head.cls_logits(F.relu(model.rpn.head.conv(out5)))
	out7 = model.rpn.head.bbox_pred(F.relu(model.rpn.head.conv(out5)))
	#roi pooling
	out8 = model.roi_heads.box_roi_pool(x,boxes,image_size).flatten(start_dim=1)
	#selecting anchor with best score to provide us best estimate of the data points required
	out8 = out8[0, :]
	out9 = F.relu(model.roi_heads.box_head.fc7(F.relu(model.roi_heads.box_head.fc6(out8))))
	#prediction
	out10 = model.roi_heads.box_predictor.cls_score(out9)
	out11 = model.roi_heads.box_predictor.bbox_pred(out9)
	
	return out0,out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11

dataset = 'voc'
def get_boxes(model, image):
	pred = model(image)
	boxes = [i["boxes"] for i in pred]
	return boxes
	
def get_featmap_and_size(model, image):
	
	out0 = image
	out1 = model.backbone[3](F.relu(model.backbone[1](model.backbone[0](out0))))
	out2 = model.backbone[4](out1)
	out3 = model.backbone[5](out2)
	out4 = model.backbone[6](out3)
	out5 = model.backbone[8](model.backbone[7](out4))
	x = OrderedDict()
	x['0'] = out5
	x['1'] = out4
	if dataset == 'kitti':
		image_size = [1200, 1200]
	elif dataset == 'coco':
		image_size = [600,600]
	else:
		image_size = [600,600]
	del out0,out1,out2,out3,out4
	return x,image_size
	

#Dataloader
dataloader = DataLoader(data, batch_size=1, shuffle=True)

for i, data in tqdm(enumerate(dataloader, 0)):
	image = data
	image = Variable(image.cuda())
	boxes = get_boxes(model, image)
	print(i)
	if (len(boxes[0]) == 0):
		continue
	else:
		x, image_size = get_featmap_and_size(model, image)
		out0,out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11 = extract_all(model, image, x, boxes, image_size)
	
	if i == 0:
		Out0 = out0.view(image.shape[0], -1).cpu().data
		Out1 = out1.view(image.shape[0], -1).cpu().data
		Out2 = out2.view(image.shape[0], -1).cpu().data
		Out3 = out3.view(image.shape[0], -1).cpu().data
		Out4 = out4.view(image.shape[0], -1).cpu().data
		Out5 = out5.view(image.shape[0], -1).cpu().data
		Out6 = out6.view(image.shape[0], -1).cpu().data
		Out7 = out7.view(image.shape[0], -1).cpu().data
		Out8 = out8.view(image.shape[0], -1).cpu().data
		Out9 = out9.view(image.shape[0], -1).cpu().data
		Out10 = out10.view(image.shape[0], -1).cpu().data
		Out11 = out11.view(image.shape[0], -1).cpu().data
	else:
		Out0 = torch.cat((Out0, out0.view(image.shape[0], -1).cpu().data), 0)
		Out1 = torch.cat((Out1, out1.view(image.shape[0], -1).cpu().data), 0)
		Out2 = torch.cat((Out2, out2.view(image.shape[0], -1).cpu().data), 0)
		Out3 = torch.cat((Out3, out3.view(image.shape[0], -1).cpu().data), 0)
		Out4 = torch.cat((Out4, out4.view(image.shape[0], -1).cpu().data), 0)
		Out5 = torch.cat((Out5, out5.view(image.shape[0], -1).cpu().data), 0)
		Out6 = torch.cat((Out6, out6.view(image.shape[0], -1).cpu().data), 0)
		Out7 = torch.cat((Out7, out7.view(image.shape[0], -1).cpu().data), 0)
		Out8 = torch.cat((Out8, out8.view(image.shape[0], -1).cpu().data), 0)
		Out9 = torch.cat((Out9, out9.view(image.shape[0], -1).cpu().data), 0)
		Out10 = torch.cat((Out10, out10.view(image.shape[0], -1).cpu().data), 0)
		Out11 = torch.cat((Out11, out11.view(image.shape[0], -1).cpu().data), 0)
	
	
print(Out0.shape)
print(Out1.shape)
print(Out2.shape)
print(Out3.shape)
print(Out4.shape)
print(Out5.shape)
print(Out6.shape)
print(Out7.shape)
print(Out8.shape)
print(Out9.shape)
print(Out10.shape)
print(Out11.shape)

torch.save(Out0, os.path.join(save_path, 'Out0'))
torch.save(Out1, os.path.join(save_path, 'Out1'))
torch.save(Out2, os.path.join(save_path, 'Out2'))
torch.save(Out3, os.path.join(save_path, 'Out3'))
torch.save(Out4, os.path.join(save_path, 'Out4'))
torch.save(Out5, os.path.join(save_path, 'Out5'))
torch.save(Out6, os.path.join(save_path, 'Out6'))
torch.save(Out7, os.path.join(save_path, 'Out7'))
torch.save(Out8, os.path.join(save_path, 'Out8'))
torch.save(Out9, os.path.join(save_path, 'Out9'))
torch.save(Out10, os.path.join(save_path, 'Out10'))
torch.save(Out11, os.path.join(save_path, 'Out11'))

del Out0,Out1,Out2,Out3,Out4,Out5,Out6,Out7,Out8,Out9,Out10,Out11


#compute ID
def computeID(r, nres, fraction):
    ID = []
    n = int(np.round(r.shape[0]*fraction))
    dist = squareform(pdist(r, 'euclidean'))
    print(dist)
    for i in range(nres):
        dist_s = dist
        perm = np.random.permutation(dist.shape[0])[0:n]
        dist_s = dist_s[perm,:]
        dist_s = dist_s[:,perm]
        ID.append(estimate(dist_s)[2])
    mean = np.mean(ID)
    error = np.std(ID)
    return mean,error
	
fname = os.path.join(save_path, 'ID_all_'+str(augmentation))
ID_all = []
method = 'euclidean'
fraction = 0.9
nres = 20

for j in tqdm(range(0,12)):
	r = torch.load(os.path.join(save_path, 'Out' + str(j)))
	ID_all.append(computeID(r, nres, fraction))
ID_all = np.array(ID_all)
np.save(fname, ID_all)
print('Final result: {}'.format(ID_all[:,0]))
print('Done')
