# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 10:51:02 2022
Detection of foot

@author: Zechen Wang
"""
import cv2
import numpy as np
def footdetector(path,filename,ksize1=(9,9),ksize2=(3,3)):
    file=path+filename
    image=cv2.imread(file)
    im_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    im_bw=cv2.GaussianBlur(im_bw, ksize1, 0)
    im_bw = cv2.Canny(im_bw, 10, 90)
    #cv2.imshow('canny',im_bw)
    #cv2.waitKey()
    im_bw=cv2.GaussianBlur(im_bw, ksize2, 0)
    
    
    
    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the index of the largest contour
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    
    return cnt

    cv2.imshow('contour',image)
    cv2.imshow('blur',im_bw)
    cv2.waitKey()
