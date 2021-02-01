import numpy as np
import os
import torch
import torchvision
from kitti_dataset import KittiLoader, AnnotationTransform_kitti
import torchvision.datasets as dset
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

root = '/home/ajay/Desktop/ID_CNN/Dataset/KITTI'
transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
  
kitti_data = KittiLoader(root=root, split='training', target_transform = AnnotationTransform_kitti())

def collate_fn(batch):
    return tuple(zip(*batch))

trainloader = DataLoader(kitti_data, batch_size=1, shuffle=True, collate_fn=collate_fn)

batch = iter(trainloader)
image = next(batch)
print(image)
