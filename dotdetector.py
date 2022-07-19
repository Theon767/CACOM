# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:19:10 2022
Library for detection of dots in images
@author: Zechen Wang
"""

import cv2
import numpy as np


def dotdetector(path_read,filename,minArea,maxArea,Convexity,Inertia):#Detect Blob features

    file=path_read+filename
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    #image = cv2.Canny(image, 10, 90) #perform Canny before blob detection
    params = cv2.SimpleBlobDetector_Params()
    
    #change thresholds
    #params.filterByColor()
    #params.minThreshold = 10;
    #params.maxThreshold = 2000;
    
    #filter by area
    params.filterByArea = True   #setting up Blob detector
    params.minArea = minArea
    params.maxArea=maxArea
    
    # Filter by Circularity(whether it is Polygon or circle)
    params.filterByCircularity = True
    params.minCircularity = 0.1
    
    # Filter by Convexity(Integrity of circle)
    params.filterByConvexity = True
    params.minConvexity = Convexity


    # Filter by Inertia(wheter it is elipse or circle)
    params.filterByInertia = True
    params.minInertiaRatio =Inertia

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)
    #cv2.imshow('canny',image)
    return keypoints
    #blank = np.zeros((1, 1))
    #blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def crosscompare(contour,keypoints,threshold=100):

    N=len(keypoints)
    new_keypoints=keypoints
    for i in range(N):
        (x,y) = keypoints[i].pt
        dst = cv2.pointPolygonTest(contour, (x,y), True) #Result is the distance between points and biggest contour
        if dst>=threshold:
            new_keypoints[i]=0 #decide whether the point is close enough to contour
    new_keypoints=[i for i in new_keypoints if i != 0] 
    return new_keypoints

def select (keypoints):#used to eliminate points which are more than 6 points,choose the left 6 points(just a Siliy function)
    N=len(keypoints)
    position=[]
    new_keypoints=keypoints

    for i in range(N):
        (x,y) = keypoints[i].pt
        position.append((x,y))
    position_sorted=sorted(position,key=lambda x:x[1])
    position=np.array(position)
    if N!=0:
        x_position=position[:,0]
    counter=-1
    while (N>6):
        position_sorted=np.delete(position_sorted,obj=-1,axis=0)
        index=sorted(enumerate(x_position), key=lambda x:x[1])
        new_keypoints[index[counter][0]]=0
        N=N-1
        counter=counter-1
    new_keypoints=[i for i in new_keypoints if i != 0]
    return new_keypoints,position_sorted
            
            