# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 12:54:52 2022

@author: m1380
"""

import os
import numpy as np
from tracker import *
from video2image import *
from angle import *
import matplotlib.pyplot as plt
path_video=r'C:\Users\m1380\CACAOM\Video'
videoname='\\10.MOV'
path_read=r'C:\Users\m1380\CACAOM\Image'
#video2image(path_video,path_read,videoname,N_frames=1)
position_sum=[]
for i in range(400,620,1): # loop over all frames
    if i<100:
        filename='\\frames_0'+str(i)+'.jpg'
    else:
        filename='\\frames_'+str(i)+'.jpg'
    file=path_read+filename
    image=cv2.imread(file)
    position,blobs,contour=tracker(path_read,filename,minArea=130,maxArea=800,Convexity=0.91,inertia=0.5,
                                   crosscompare_imp=True,ksize1=(3,3),ksize2=(5,5),dst_threshold=500, 
                                   select_imp=True)
    #cv2.imshow(filename,blobs)
    print(len(position))
    if len(position)==6:
        if isinstance(position,list)==True:
            position=np.asarray(position)
        position_sum.append(position)
    #print(position)
    #image=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #print(image[round(position[0][0]-100),round(position[0][1]-100)])
    #print(image[round(position[1][0]),round(position[1][1])])
    #print(image[round(position[2][0]),round(position[2][1])])
    #print(image[round(position[3][0]),round(position[3][1])])
    #cv2.drawContours(image,contour, -1, (0,0,255), 3) #Draw biggest contour
    #cv2.imshow('contour',image)
    #cv2.waitKey()
position_sum=np.asanyarray(position_sum)
new_position=position_sum[:,0:4,:]
final_position=[]
for i in range(len(new_position)):
    pos=np.sort(new_position[i,:,:],axis=0)[[0,2,3],:]
    final_position.append(pos)
final_position=np.asarray(final_position)

angle_sum=[]
for i in range(len(final_position)):
    angle=return_angle(final_position[i,:,:])
    angle_sum.append(angle)
angle_sum=np.asarray(angle_sum)

x=range(len(angle_sum))
plt.xlabel('Frames')
plt.ylabel('Foot arch angle')
plt.plot(x,angle_sum)
plt.show()

