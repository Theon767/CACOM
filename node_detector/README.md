
# README.md

This documentation will guide you how to run this `node_detector.py` in your own local 
environment step by step.  

## Abstract
As mentioned before, the main goal of this project is to design a user-friendly programmm
for the people, who want to know if their gait is normal or not, but don't want to give their
time to doctor. Based on this purpose, a very convinient __VISION-based__ tool is designed. With its 
help, people are able to just take the video by themselves, and do some small processinng work, and wait
for just few minutes to obtain the result. In this way, people will save their time significantly.  

In this project, all parts are written in language `PYTHON`.

## Preknowledge - What should we have before running this script

```
|
|- Videos taken by the user or images transformed from the video
|- All blobs detected from each video frame (or from every image)
|- Annotated images in MIVOS (Optional, for model fine-tuning)

```

## To run this code, you have to...
### 0) have a graphic card with memory at least 2GB for CNN processing.

### 1) Set up the environment for the DNN model. (Copyright by @Yuankai_Wu)

 - Install __Detectron2__ dependencies.
 ```
 # If you are using terminal, remove "!" at first.
 !pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
 !pip install git+https://github.com/facebookresearch/fvcore.git
 ```
 - Download __Detectron2__ from facebook research
 ```
 !git clone https://github.com/facebookresearch/detectron2 detectron2_repo
 ```
 - Install __Detectron2__

 ```
 !pip install -e detectron2_repo
 #!pip install detectron2
 !pip install 'git+https://github.com/facebookresearch/detectron2.git'
 !pip install pycocotools
 ```
Congrats! The __Detectron2__ is ready for use!

### 2) Modify paths
- Bad news: This is the most annoying step :(((
- Good News: After this part the code runs just like butter! :D
#### Here you have to modify the following paths:
- `test_path`: Must be a folder, which contains all images extracted from the recorded video
- `cfg_path`: Default path to configuration file for __Detectron2__
- `dataset_path`: Must be a folder, which contains all things related to model training and __Detectron2Wrapper__ initializing
- `model_path`: The __MOST IMPORTANT ONE__, Default path to the __pretrained__ model.
You will find those parameters on the most top of code.  
All files mentioned above you can find in the drive folder. :)

### 3) Parametrize your analysis (Optional, but highly recommanded)
In this step you will change soem parameters based on your __own video__. You can find this part in the beginning of
`if __name__ == "__main__:"`
### 4) Execution. It's time to press "ctrl" + "enter"!
After several minutes processing time, you will get a list named `markerAllframe`. If in the previous step you decided to save the output
to `.CSV` file, this list will be written in a `.CSV` file with the name you gave (also in the previous step.)
## Author

- [@Shuang_Wang] Matrikelnummer/Registrition number: 03735770
