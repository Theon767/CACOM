"""
Created on Tue Jun 21 09:19:10 2022
Used for transformation from video to images
@author:Zechen Wang
"""
import cv2
import os
def video2image(path_read,path_write,filename,N_frames):
    #path_read=r'C:\Users\m1380\CACAOM\Video\IMG_3969.MOV'
    #path_read1=r'C:\Users\m1380\CACAOM\Video'
    #path_write=r'C:\Users\m1380\CACAOM\Image'
    file=path_read+filename
    vidcap = cv2.VideoCapture(file)
    #fps= vidcap.get(cv2.CAP_PROP_FPS)
    #print('fps:',fps)
    success=1
    frame_count=1   #+1 for each iteration 
    #N_frames=10     #fetch picture every N frames
    while success:
        os.chdir(path_read)
        success,image=vidcap.read()
        if not success:
            print("Process finished")
            break
        if frame_count % N_frames ==0:
            image_name='frames_'+str(frame_count).rjust(3,'0')+".jpg"
            os.chdir(path_write)
            cv2.imwrite(image_name,image)     # save frame as JPEG file      
            print('Read a new frame: ')
        frame_count += 1