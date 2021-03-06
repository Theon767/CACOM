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
def tracker(path_read,filename,crosscompare_imp=True,minArea=110,maxArea=500,ksize1=(9,9),ksize2=(3,3),Convexity=0.93,inertia=0.1,select_imp=True,dst_threshold=100):
    position=[] #if no features then defalut position is null
    file=path_read+filename#load the file
    image=cv2.imread(file)
    keypoints=dotdetector(path_read,filename,minArea,maxArea,Convexity,inertia)#using blob to extract feature points,minArea suggest the size 
                                                                            #of the blob features
    
    max_contour=footdetector(path_read,filename,ksize1,ksize2)#using Canny&Contour to find the contour of feet 
    
    if crosscompare_imp==False:# Decide whether to implement crosscompare of Blob(dot)detector and footdetector
        new_keypoints=keypoints      
    else:                                                                       #new_keypoints=keypoints
        new_keypoints=crosscompare(max_contour,keypoints,dst_threshold)  # eliminate feature points which are not on feet

    #detect foot and Blob features and ensure that features are on foot
    
    if select_imp==True: # decide whther to limit the number of feature points and return position(occasionally they are larger than 6)
        new_keypoints,position=select(new_keypoints)
    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(image, new_keypoints, blank, (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) #the image with blob feature on it
    #cv2.drawContours(image,max_contour, -1, (0,255,0), 3) #Draw biggest contour
    
    
    return position,blobs,max_contour