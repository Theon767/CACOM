#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 16:45:24 2022

For extracting frames from samples in the video form,
change the  v_path  to the path of the video 
       and  i_path  to the folder where the images are going to.
 
       i_path is not neccessarily to be an existing folder
       if the path does not exist, it will be created automatically.
        
@author: kolz14w
"""


from os import listdir, makedirs, getcwd
from os.path import join, isdir
import cv2 as cv
import shutil

v_path = "/home/kolz14w/下载/2022_07_06_14_03_IMG_4265.MOV"
i_path = "/home/kolz14w/桌面/frames"

def img_normalizer(img):
    standard_dim = [853, 480]
    img_dim = img.shape
    if ~(standard_dim == img_dim):
        new_img = cv.resize(img, standard_dim, interpolation = cv.INTER_AREA)
    else:
        new_img = img
    return new_img

def vid_2_img(v_path, i_path):
    
    # generate frames in the current folder
    vc = cv.VideoCapture(v_path)
    c = 0
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        print('Openerror! ')
        rval = False
    num_frame = int(vc.get(7))       # get the total number of frames in the selected video
    for i in range(num_frame-1):
        rval, frame = vc.read()
        new_frame = img_normalizer(frame)   # resize frames to 853, 480 in order to fit mask-rcnn
        c += 1
        cv.imwrite('frame%04d.jpg'%i, new_frame)
        
    # move frames to the target folder
    current_path = getcwd()
    for item in listdir(current_path):
        if item.endswith('.jpg'):
            shutil.move(item, i_path)
    vc.release()
    print("unlock image: ", c)
        
if __name__ == "__main__":
    if not isdir(i_path):
        makedirs(i_path)
    vid_2_img(v_path, i_path)
