import os
import numpy as np
import torch
import collections
from PIL import Image
import torch.utils.data
import xml.etree.ElementTree as ET

VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor']

class VOC(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "Annotations"))))
        assert (len(self.imgs) == len(self.labels))
       

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
        label_path = os.path.join(self.root, "Annotations", self.labels[idx])
        image = Image.open(img_path).convert("RGB")
       
        target = self.parse_voc_xml(ET.parse(label_path).getroot())
        targets = {}
        num_objs = len(target['annotation']['object'])
        labels, boxes, areas = [], [], []
        for i in range(num_objs):
            label = (target['annotation']['object'][i]['name'])
            labels.append(VOC_CLASSES.index(label))
            xmin = int(target['annotation']['object'][i]['bndbox']['xmin'])
            xmax = int(target['annotation']['object'][i]['bndbox']['xmax'])
            ymin = int(target['annotation']['object'][i]['bndbox']['ymin'])
            ymax = int(target['annotation']['object'][i]['bndbox']['ymax'])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.int64)
        
        img_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:,2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs, ), dtype=torch.int64)   

        targets["boxes"] = boxes
        targets["labels"] = labels
        targets["img_id"] = img_id
        targets["area"] = area
        targets["iscrowd"] = iscrowd
        
        
        if self.transforms is not None:
            image, targets = self.transforms(image, targets)
        
        return image, targets
    
    def __len__(self):
        return(len(self.imgs))

    def parse_voc_xml(self, node: ET.Element):
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == 'annotation':
                def_dic['object'] = [def_dic['object']]
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict



