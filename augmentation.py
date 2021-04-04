

import torch
import cv2
import os
import random
import numpy as np

def fill(img, h, w):
    img = cv2.resize(img, (h,w), cv2.INTER_CUBIC)
    return img

def kitti_horizontal_shift(img, ratio=0.0):
    if ratio>1 or ratio<0:
        print("Value should be less than 1 and greater than 0")
        return img
    ratio = random.uniform(-ratio, ratio)
    w, h = img.shape[:2]
    to_shift = h*ratio
    if ratio>0:
        img = img[:, :int(h-to_shift), : ]
    if ratio<0:
        img = img[:, int(-1*to_shift):, : ]
    img = fill(img, h, w)
    return img

def kitti_vertical_shift(img, ratio=0.0):
    if ratio>1 or ratio<0:
        print("Value should be less than 1 and greater than 0")
        return img
    ratio = random.uniform(-ratio, ratio)
    w, h = img.shape[:2]
    to_shift = w*ratio
    if ratio>0:
        img = img[:int(w-to_shift), :, : ]
    if ratio<0:
        img = img[int(-1*to_shift):, :, : ]
    img = fill(img, h, w)
    return img

def horizontal_shift(img, ratio=0.0):
    if ratio>1 or ratio<0:
        print("Value should be less than 1 and greater than 0")
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = w*ratio
    if ratio>0:
        img = img[:, :int(w-to_shift), : ]
    if ratio<0:
        img = img[:, int(-1*to_shift):, : ]
    img = fill(img, h, w)
    return img

def vertical_shift(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = h*ratio
    if ratio > 0:
        img = img[:int(h-to_shift), :, :]
    if ratio < 0:
        img = img[int(-1*to_shift):, :, :]
    img = fill(img, h, w)
    return img

def channel_shift(img, value):
    value = int(random.uniform(-value, value))
    img = img + value
    img[:,:,:][img[:,:,:]>255]  = 255
    img[:,:,:][img[:,:,:]<0]  = 0
    img = img.astype(np.uint8)
    return img

def horizontal_flip(img, flag):
    if flag:
        return cv2.flip(img, 1)
    else:
        return img

def vertical_flip(img, flag):
    if flag:
        return cv2.flip(img, 0)
    else:
        return img

def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

def create_augmentation(path):
    base_path = os.path.split(path)
    #Horizontal Shift
    save_path = os.path.join(base_path[0], "horizontal_shift")
    if not os.path.exists(save_path):
        os.mkdir(os.path.join(base_path[0], "horizontal_shift"))
    os.chdir(save_path)
    print("Horizontal Shift in Progress...")
    for image in os.listdir(path):
        img = cv2.imread(os.path.join(path, str(image)))
        if path == KITTI_PATH:
            img = kitti_horizontal_shift(img, 0.7)
        else:
            img = horizontal_shift(img, 0.7)
        cv2.imwrite(image, img)
    #Vertical Shift
    save_path = os.path.join(base_path[0], "vertical_shift")
    if not os.path.exists(save_path):
        os.mkdir(os.path.join(base_path[0], "vertical_shift"))
    os.chdir(save_path)
    print("Vertical Shift in Progress...")
    for image in os.listdir(path):
        img = cv2.imread(os.path.join(path, str(image)))
        if path == KITTI_PATH:
            img = kitti_vertical_shift(img, 0.7)
        else:
            img = vertical_shift(img, 0.7)
        cv2.imwrite(image, img)
    #Channel Shift
    save_path = os.path.join(base_path[0], "channel_shift")
    if not os.path.exists(save_path):
        os.mkdir(os.path.join(base_path[0], "channel_shift"))
    os.chdir(save_path)
    print("Channel Shift in Progress...")
    for image in os.listdir(path):
        img = cv2.imread(os.path.join(path, str(image)))
        img = channel_shift(img, 50)
        cv2.imwrite(image, img)
    #Horizontal Flip
    print("Horizontal Flip in progress...")
    save_path = os.path.join(base_path[0], "horizontal_flip")
    if not os.path.exists(save_path):
        os.mkdir(os.path.join(base_path[0], "horizontal_flip"))
    os.chdir(save_path)
    for image in os.listdir(path):
        img = cv2.imread(os.path.join(path, str(image)))
        img = horizontal_flip(img, True)
        cv2.imwrite(image, img)
    #Vertical Flip
    print("Vertical Flip in progress...")
    save_path = os.path.join(base_path[0], "vertical_flip")
    if not os.path.exists(save_path):
        os.mkdir(os.path.join(base_path[0], "vertical_flip"))
    os.chdir(save_path)
    for image in os.listdir(path):
        img = cv2.imread(os.path.join(path, str(image)))
        img = vertical_flip(img, True)
        cv2.imwrite(image, img)
    #Rotation
    print("Rotation in progress...")
    save_path = os.path.join(base_path[0], "rotation")
    if not os.path.exists(save_path):
        os.mkdir(os.path.join(base_path[0], "rotation"))
    os.chdir(save_path)
    for image in os.listdir(path):
        img = cv2.imread(os.path.join(path, str(image)))
        img = rotation(img, 45)
        cv2.imwrite(image, img)
    print("Augmentation COMPLETE!!!")


def main():
KITTI_PATH = "/work/chawda/Dataset/KITTI/testing/image_2"
COCO_PATH = "/work/chawda/Dataset/COCO/test2017"
VOC_PATH = "/work/chawda/Dataset/VOCdevkit/VOC2007/JPEGImages"

create_augmentation(KITTI_PATH)
create_augmentation(COCO_PATH)
create_augmentation(VOC_PATH)

if __name__ == '__main__':
main()

