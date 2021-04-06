#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd


KITTI_PATH = "D:/Dataset/KITTI/testing/image_2"
COCO_PATH = "D:/Dataset/COCO/test2017"
VOC_PATH = "D:/Dataset/VOCdevkit/VOC2007/JPEGImages"



def calculate_FD(image, size):
    blue_channel = image[:,:,0]
    green_channel = image[:,:,1]
    red_channel = image[:,:,2]
    
    red = red_channel.flatten()
    green = green_channel.flatten()
    blue = blue_channel.flatten()
    
    red_max = np.max(red)
    green_max = np.max(green)
    blue_max = np.max(blue)
    
    M = size
    q = 0.21*float(red_max) + 0.72*float(green_max) + 0.07*float(blue_max)
    q = np.ceil(q)
    scale = []
    lr =[]
    l = 2
    Nr = []
    scale = []
    
    while l <= (M/2):
        r = l
        slice_number = 1
   
        ld = (l*q)/M
        nr = 0
        blocksize_row = r
        blocksize_col = r
        for row in range(1,M, blocksize_row):
            for col in range(1,M, blocksize_col):
                row_1 = row
                row_2 = row_1 + blocksize_row - 1
                col_1 = col
                col_2 = col_1 + blocksize_col - 1
                one_block = image[row_1:row_2, col_1:col_2]
                max_intensity = np.sum(np.max(np.max(one_block)))/3
                min_intensity = np.sum(np.min(np.min(one_block)))/3
           
                height_box = np.ceil(float(max_intensity) / ld)
                if max_intensity == min_intensity:
                    nr = nr + 1
                else:
                    nr = nr + height_box
                slice_number = slice_number + 1
        Nr.append(nr)
        scale.append(M/l)
        l = l*2
    N = np.log(Nr)
    S = np.log(scale)
    p = np.polyfit(S, N, 1)
    f = np.polyval(p,S)
    #print("Fractal Dimension is", p[0])
    m = p[0]
    c = p[1]
    y = 0
    for j in range(1, len(N)):
        x = (((m * S[j]) + c) - N[j])/(1 + m**2)
        if x < 0:
            y = y + x * (-1)
        else:
            y = y + x
    E = (1/len(N))*np.sqrt(y)
    #print("Error is", E)
    result = [p[0], E]
    return result


def store_results(path, directory):
    if path == KITTI_PATH:
        size = 1200
    else:
        size = 300
    base_path = os.path.split(path)
    #Horizontal Shift
    save_path = os.path.join(base_path[0], directory)
    image = cv2.imread(save_path)
    FD = []
    for image in os.listdir(save_path):
        img = cv2.imread(os.path.join(save_path, str(image)))
        dim = calculate_FD(img, size)
        FD.append(dim) 
    df = pd.DataFrame(FD, columns=['Dimension', 'Error'])
    return df

def main():
    directory = ['horizontal_shift', 'vertical_shift', 'horizontal_flip', 'vertical_flip', 'rotation', 'channel_shift' ]
    for i in range(0, len(directory)):
        print("KITTI", directory[i])
        data_kitti[i] = store_results(KITTI_PATH, directory[i])
        print("COCO", directory[i])
        data_coco[i] = store_results(COCO_PATH, directory[i])
        print("VOC", directory[i])
        data_voc[i] = store_results(VOC_PATH, directory[i]) 
    directory = ['horizontal_shift', 'vertical_shift', 'horizontal_flip', 'vertical_flip', 'rotation', 'channel_shift' ]
    for i in range(0, len(directory)):
        if i == 0:
            kitti_horizontal_shift = data_kitti[i]
            coco_horizontal_shift = data_coco[i]
            voc_horizontal_shift = data_voc[i]
            kitti_horizontal_shift.to_csv()
            coco_horizontal_shift.to_csv()
            voc_horizontal_shift.to_csv()
        elif i == 1:
            kitti_vertical_shift = data_kitti[i]
            coco_vertical_shift = data_coco[i]
            voc_vertical_shift = data_voc[i]
            kitti_vertical_shift.to_csv()
            coco_vertical_shift.to_csv()
            voc_vertical_shift.to_csv()
        elif i == 2:
            kitti_horizontal_flip = data_kitti[i]
            coco_horizontal_flip = data_coco[i]
            voc_horizontal_flip = data_voc[i]
            kitti_horizontal_flip.to_csv()
            coco_horizontal_flip.to_csv()
            voc_horizontal_flip.to_csv()
        elif i == 3:
            kitti_vertical_flip = data_kitti[i]
            coco_vertical_flip = data_coco[i]
            voc_vertical_flip = data_voc[i]
            kitti_vertical_flip.to_csv()
            coco_vertical_flip.to_csv()
            voc_vertical_flip.to_csv()
        elif i == 4:
            kitti_rotation = data_kitti[i]
            coco_rotation = data_coco[i]
            voc_rotation = data_voc[i]
            kitti_rotation.to_csv()
            coco_rotation.to_csv()
            voc_rotation.to_csv()
        else:
            kitti_channel_shift = data_kitti[i]
            coco_channel_shift = data_coco[i]
            voc_channel_shift = data_voc[i]
            kitti_channel_shift.to_csv()
            coco_channel_shift.to_csv()
            voc_channel_shift.to_csv()  


if __name__ == '__main__':
    main()   

