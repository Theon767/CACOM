#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 10:43:20 2022

@author: kolz14w
"""

import cv2
import os

image_folder = '/home/kolz14w/桌面/preimg'
video_name = "test_vid.avi"

images = [img for img in os.listdir(image_folder) if img.endswith('jpg')]
reordered_images = []
idx_original = []

for image in images:
    frame_idx = int(image[-8: -4])
    idx_original.append(frame_idx)
    
for i in range(len(idx_original)):
    idx_current_frame = idx_original.index(min(idx_original))
    current_frame = images[idx_current_frame]
    reordered_images.append(current_frame)
    idx_original[idx_current_frame] = 10000
    
frame = cv2.imread(os.path.join(image_folder, reordered_images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 30, (width, height))

for image in reordered_images:
    video.write(cv2.imread(os.path.join(image_folder, image)))
    
cv2.destroyAllWindows()
video.release()