# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:19:10 2022
Library for detection of dots in images
@author: Zechen Wang
"""

import cv2

def footdetector(path_read,filename,threshold):
    #threshold=10
    #path_read=r'C:\Users\m1380\CACAOM\Image'
    file=path_read+filename
    image = cv2.imread(file) #Detect the coordinates of foot
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    shifted=hsv[:,:,0]
    shifted[shifted>threshold]=127
    return shifted

def dotdetector(path_read,filename,minArea):#Detect Blob features
    #path_read=r'C:\Users\m1380\CACAOM\Image'
    #defalut minArea=70
    file=path_read+filename
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True   #setting up Blob detector
    params.minArea = minArea
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)
    return keypoints
    #blank = np.zeros((1, 1))
    #blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def crosscompare(foot_image,keypoints,size):
    N=len(keypoints) #used to ensure that Blob features lie on foot
    print(N)
    new_keypoints=keypoints
    for i in range(N):
        #x=list(range(round(keypoints[i].pt[0])-20,round(keypoints[i].pt[0])+20)) 
        #y=list(range(round(keypoints[i].pt[1])-20,round(keypoints[i].pt[1])+20))
        #print(i)
        (x,y) = keypoints[i].pt
        #print(keypoints[0].pt)
        #print(keypoints[1].pt)
        #print(keypoints[2].pt)
        #print(keypoints[3].pt)
        a=foot_image[round(x)-size:round(x)+size#form a square area around Blob features
                      ,round(y)-size:round(y)+size]
        if (a==127).any():
            new_keypoints[i]=0
            #print('yes')
            #print(keypoints)
    new_keypoints= [i for i in new_keypoints if i != 0]
    return new_keypoints
            
            