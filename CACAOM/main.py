# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 15:22:50 2022

@author: Zechen Wang
"""
import os
import cv2
import numpy as np
from dotdetector import *
from video2image import *
path_read=r'C:\Users\m1380\CACAOM\Video'
path_write=r'C:\Users\m1380\CACAOM\Image'
filename='\\frames_060.jpg'
video2image(path_read,path_write,'\\IMG_3982.MOV',10)
file=path_write+filename
print(file)
image=cv2.imread(file)
#load the file

foot_image=footdetector(path_write,filename,threshold=10)
keypoints=dotdetector(path_write,filename,minArea=50)
#new_keypoints=

new_keypoints=crosscompare(foot_image,keypoints,size=0)
#new_keypoints=keypoints
#detect foot and Blob features and ensure that features are on foot


blank = np.zeros((1, 1))
blobs = cv2.drawKeypoints(image, new_keypoints, blank, (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('blobs',blobs)
cv2.waitKey()
cv2.imshow('blobs2',foot_image)
cv2.waitKey()