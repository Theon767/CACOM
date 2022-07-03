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
from footdetector import *
    
path_read=r'C:\Users\m1380\CACAOM\Video'
path_write=r'C:\Users\m1380\CACAOM\Image'
filename='\\frames_280.jpg'
#video2image(path_read,path_write,'\\IMG_3998.MOV',10)# transform video to picture
file=path_write+filename#load the file
image=cv2.imread(file)


keypoints=dotdetector(path_write,filename,minArea=60)#using blob to extract feature points,minArea suggest the size 
                                                                        #of the blob features

max_contour=footdetector(path_write,filename,ksize1=(9,9),ksize2=(3,3))#using Canny&Contour to find the contour of 
                                                                            #feet 

new_keypoints=crosscompare(max_contour,keypoints)  # eliminate feature points which are not on feet
#new_keypoints=keypoints
#detect foot and Blob features and ensure that features are on foot


blank = np.zeros((1, 1))
blobs = cv2.drawKeypoints(image, new_keypoints, blank, (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.drawContours(image,max_contour, -1, (0,255,0), 3) #Draw biggest contour

cv2.imshow('blobs2',image)
cv2.waitKey()

cv2.imshow('blobs',blobs)
cv2.waitKey()