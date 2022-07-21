#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 10:43:20 2022

@author: kolz14w
"""

import cv2
import os

image_folder = 'images'
video_name = "test_vid.avi"

images = [img for img in os.listdir(image_folder) if img.endwith('jpg')]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))
    
cv2.destroyAllWindows()
video.release()