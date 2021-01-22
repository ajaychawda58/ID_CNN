import os

path = "/home/ajay/Desktop/Dataset/VOCdevkit/VOC2012/Annotations/"

with open('ann_path.txt', 'w') as f:
    for file in os.listdir(path):
        data = path + file + "\n"
        f.write(data)



