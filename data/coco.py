#https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
from PIL import Image
import os
import os.path
from typing import Any, Callable, Optional, Tuple
import torch.utils.data
import torch
import torchvision
from pycocotools.coco import COCO


class Coco(torch.utils.data.Dataset):
   

    def __init__(self, root, annotations, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotations)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        num_objs = len(target)

        boxes, labels, areas, iscrowd = [], [], [], []
        for i in range(num_objs):
            xmin = target[i]['bbox'][0]
            ymin = target[i]['bbox'][1]
            xmax = xmin + target[i]['bbox'][2]
            ymax = ymin + target[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            areas.append(target[i]['area'])
            labels.append(target[i]['category_id'])
            iscrowd.append(target[i]['iscrowd'])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        img_id = torch.tensor([img_id])
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        targets = {}
        targets['boxes'] = boxes
        targets['labels'] = labels
        targets['image_id'] = img_id
        targets['area'] = areas
        targets['iscrowd'] = iscrowd
        target = targets
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target


    def __len__(self) -> int:
        return len(self.ids)
