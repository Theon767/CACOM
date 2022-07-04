# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 17:46:05 2022

@author: m1380
"""
import cv2
import os

path_write=r'C:\Users\m1380\CACAOM\Image'
filename='\\frames_060.jpg'
file=path_write+filename
image=cv2.imread(file)
b,g,r=cv2.split(image)
cv2.imshow('blue',b)
cv2.waitKey()