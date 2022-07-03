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

v_path = "/home/kolz14w/下载/IMG_4816.MOV"
i_path = "/home/kolz14w/桌面/frames"

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
        c += 1
        cv.imwrite('frame'+str(c)+'.jpg', frame)
        
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
