# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 12:54:52 2022

@author: m1380
"""

import os
import numpy as np
from tracker import *
from video2image import *
path_read=r'C:\Users\m1380\CACAOM\Image'
for i in range(10,200,10): # loop over all frames
    if i<100:
        filename='\\frames_0'+str(i)+'.jpg'
    else:
        filename='\\frames_'+str(i)+'.jpg'
    file=path_read+filename
    image=cv2.imread(file)
    
    
    position,blobs,contour=tracker(path_read,filename,minArea=130,maxArea=800,Convexity=0.91,inertia=0.5,
                                   crosscompare_imp=True,ksize1=(3,3),ksize2=(5,5),dst_threshold=10,
                                   select_imp=True)
    cv2.imshow(filename,blobs)
    print(filename)
    print(position)
    #cv2.drawContours(image,contour, -1, (0,0,255), 3) #Draw biggest contour
    #cv2.imshow('contour',image)

cv2.waitKey()

filename='\\frames_300.jpg'