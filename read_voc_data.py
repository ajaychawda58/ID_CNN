
import numpy as np
import os
import torch
import torchvision
import torchvision.datasets as dset
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

root = '/home/ajay/Desktop/ID_CNN/Dataset/VOCdata'
transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

voc_data = dset.VOCDetection(root=root, year='2012', image_set='train', download=False, transform = transform)

def collate_fn(batch):
    return tuple(zip(*batch))

trainloader = DataLoader(voc_data, batch_size=32, shuffle=True, collate_fn=collate_fn)

image, target = next(iter(trainloader))
print(len(image), len(target))
