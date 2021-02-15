#https://github.com/abpaudel/faster-rcnn/blob/master/Faster_R_CNN.ipynb

import os
from PIL import Image
import numpy as np
import torch
import torch.utils.data

KITTI_CLASSES = ['BG', 'Car', 'Van', 'Truck', 'Pedestrian',
                'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']


class KITTI(object):
    def __init__(self, path, transforms=None):
        self.path = path
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(path, "training/image_2"))))
        self.labels = list(sorted(os.listdir(os.path.join(path, "training/label_2"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.path, "training/image_2", self.imgs[idx])
        label_path = os.path.join(self.path, "training/label_2", self.labels[idx])
        image = Image.open(img_path).convert("RGB")
        
        boxes, label = [], []
        objs = 0
        for i in open(label_path).readlines():
            i = i.split()
            label.append(KITTI_CLASSES.index(i[0]))
            boxes.append(list(map(float, i[4:8])))
            objs += 1
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        label = torch.as_tensor(label, dtype=torch.int64)
        img_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:,2] - boxes[:, 0])
        iscrowd = torch.zeros((objs, ), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = label
        target["image_id"] = img_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self):
        return len(self.imgs)

    
