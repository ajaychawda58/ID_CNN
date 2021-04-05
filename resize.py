#!/usr/bin/env python
# coding: utf-8

import cv2
import os
import numpy as np


KITTI_PATH = "/work/chawda/Dataset/KITTI/testing/image_2"
COCO_PATH = "/work/chawda/Dataset/COCO/test2017"
VOC_PATH = "/work/chawda/Dataset/VOCdevkit/VOC2007/JPEGImages"


def resize(img, h, w):
    img = cv2.resize(img, (h,w))
    return img


def resize_image(path, directory):
    base_path = os.path.split(path)
    #Horizontal Shift
    save_path = os.path.join(base_path[0], directory)
    if not os.path.exists(save_path):
        print("Folder does not exist")
    os.chdir(save_path)
    print("Resize in Progress...")
    count = 0
    for image in os.listdir(save_path):
        if count > 5000:
            os.remove(os.path.join(save_path, str(image)))
        else:
            img = cv2.imread(os.path.join(save_path, str(image)))
            if path == KITTI_PATH:
                img = resize(img, 1200, 1200)
            elif path == COCO_PATH:
                img = resize(img, 300, 300)
            else:
                img = resize(img, 300, 300)
            cv2.imwrite(image, img)
        count += 1


def main():
    directory = ['horizontal_shift', 'vertical_shift', 'horizontal_flip', 'vertical_flip', 'rotation', 'channel_shift' ]
    for i in range(0, len(directory)):
        resize_image(KITTI_PATH, directory[i])
        resize_image(COCO_PATH, directory[i])
        resize_image(VOC_PATH, directory[i])       


if __name__ == '__main__':
    main()







