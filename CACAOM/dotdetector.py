# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:19:10 2022
Library for detection of dots in images
@author: Zechen Wang
"""

import cv2
import numpy as np


def dotdetector(path_read,filename,minArea):#Detect Blob features
    #path_read=r'C:\Users\m1380\CACAOM\Image'
    #defalut minArea=70
    file=path_read+filename
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
    params = cv2.SimpleBlobDetector_Params()
    
    #change thresholds
#   params.minThreshold = 10;
#    params.maxThreshold = 200;
    
    #filter by area
    params.filterByArea = True   #setting up Blob detector
    params.minArea = minArea
    
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.3
   

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.2

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)
    return keypoints
    #blank = np.zeros((1, 1))
    #blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def crosscompare(contour,keypoints):

    N=len(keypoints)
    new_keypoints=keypoints
    for i in range(N):
        (x,y) = keypoints[i].pt
        result = cv2.pointPolygonTest(contour, (x,y), False)
        if result==1 or result==0:
            new_keypoints[i]=0
    new_keypoints=[i for i in new_keypoints if i != 0]
    return new_keypoints

def select (keypoints):
    N=len(keypoints)
    position=[]
    new_keypoints=keypoints

    for i in range(N):
        (x,y) = keypoints[i].pt
        position.append((x,y))
    if N>6:
        position=sorted(position,key=lambda x:x[1])
        position=np.delete(position,obj=-1,axis=0)
    return position
            
            