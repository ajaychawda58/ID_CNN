import torch
import read_data
import torchvision
from read_data import PascalVOC

train_data_dir = '/home/ajay/Desktop/ID_CNN/Dataset/VOCdevkit/JPEGImages'
train_voc = '/home/ajay/Desktop/ID_CNN/Dataset/VOCdevkit/VOC2012/new_labels.json'



def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)

def collate_fn(batch):
    return(tuple(zip(*batch)))
voc_data = PascalVOC(root=train_data_dir, annotation = train_voc, transforms=get_transform())

data_loader = torch.utils.data.DataLoader(voc_data, batch_size = 10, shuffle=True, collate_fn=collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# DataLoader is iterable over Dataset
for imgs, annotations in data_loader:
    imgs = list(img.to(device) for img in imgs)
    annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
    print(annotations)
