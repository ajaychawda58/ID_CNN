import os
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from torchvision import utils, transforms
from pycocotools.coco import COCO

root = '/home/ajay/Desktop/ID_CNN/Dataset/COCO/train_2017/train2017/'
annotations = '/home/ajay/Desktop/ID_CNN/Dataset/COCO/annotations_COCO/annotations/instances_train2017.json'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
coco = dset.CocoDetection(root = root, annFile=annotations, transform=transform)

train = DataLoader(coco, batch_size=32, shuffle=True)

print(len(coco))

images, target = next(iter(train))

print(len(images))

