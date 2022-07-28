# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from matplotlib import pyplot as plt
from functools import partial
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import torch, torchvision
import numpy as np
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
import pycocotools.mask as coco_mask
from detectron2.config import get_cfg
import os

# Visualizer packages
from matplotlib import pyplot as plt
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode


# Initialize the configuration of the model output to yaml file
# Before using this function, make sure that the dataset is already
# successfully registered in class DatasetCatalog, otherwise could 
# cause error.
# All parameters in this cfg initialization function can be modified
# to fit the custom scenario

def Init_Save_Cfg(**kwargs):
    # initialize some deflaut paths in case that there
    # is no argument given
    default_cfg_path = "/home/kolz14w/Proj_Master/haucode/hw2/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    default_model_path = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"    
    default_num_categories = 2
    cfg = get_cfg()
    
    if 'cfg_path' in kwargs.keys():
        cfg.merge_from_file(kwargs['cfg_path'])
    else:
        cfg.merge_from_file(default_cfg_path) 
    #Train/test set assertion
    cfg.DATASETS.TRAIN = ("Mivos",)
    cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.DEVICE= 'cuda:0'   #for GPU, otherwise 'cpu'
    # initialize the model weights with custom weights or pretrained weights
    if 'model_path' in kwargs.keys():
        cfg.MODEL.WEIGHTS = kwargs['model_path']
    else:
        cfg.MODEL.WEIGHTS = default_model_path
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0005
    cfg.SOLVER.MAX_ITER = 1000 # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64   # faster, and good enough for this toy dataset
    if 'num_categories' in kwargs.keys():
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = kwargs['num_categories']
    else:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = default_num_categories
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    
    # save configuration unter current folder
    with open('cfgCAOCM.yaml', 'w') as f:
        f.write(cfg.dump())
    
    # if you want to merge cfg from other cfg without reading os path, the output of
    # this function can also be used
    return cfg

def img_normalizer(img):
    standard_dim = [853, 480]
    img_dim = img.shape
    if ~(standard_dim == img_dim):
        new_img = cv2.resize(img, standard_dim, interpolation = cv2.INTER_AREA)
    else:
        new_img = img
    return new_img
        
# Visualizer initialization, argument is the class Detectron2Wrapper

def init_visualizer(det, test_path = 'mask_rcnn/Mivos_cus/images/frame0005.jpg'):    
    img_test = cv2.imread(test_path)
    
    result = det.predictor(img_test)
    
    v = Visualizer(img_test[:, :, ::-1],
                       metadata=det.meta, 
                       scale=0.8, 
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
    
    v = v.draw_instance_predictions(result["instances"].to("cpu"))
    
    plt.figure()
    
    plt.imshow(v.get_image()[:, :, ::-1])
        
# Create object masks
# for custom setting got masks 
# 1.red - points 
    # 2.green - bowl 
    # 3.purpue - Oats 
    # 4.yellow - cup
    # 5. blue - spoon 
    # 6.bright blue - hand

def create_sub_masks(mask_image, width, height):
    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}

    # 'r': 1.0, 'g': 0.9215686321258545, 'b': 0.01568627543747425, 'a': 1.0

    # sub_masks[1] = mask_image[:,:,1.0, 0.9215686321258545, 0.01568627543747425]
    # sub_masks[1] = mask_image[:,:]
    img1 = np.zeros((width, height), dtype=np.uint8)
    # img2 = np.zeros((width, height), dtype=np.uint8)
    # img3 = np.zeros((width, height), dtype=np.uint8)
    # img4 = np.zeros((width, height), dtype=np.uint8)
    # img5 = np.zeros((width, height), dtype=np.uint8)
    # img6 = np.zeros((width, height), dtype=np.uint8)

    for x in range(width):
        for y in range(height):
            pixel = mask_image[x, y]
            if pixel[2] == (int)(128) and pixel[1] == (int)(0) and pixel[0] == (int)(0):
                img1[x, y] = 255        # [128, 128, 0] - bright blue - Hand
            # elif pixel[2] == 0 and pixel[1] == (int)(128) and pixel[0] == (int)(128):
            #     img2[x, y] = 255        # [0, 0, 128] - red - Milk
            # elif pixel[2] == (int)(128) and pixel[1] == (int)(128) and pixel[0] == (int)(0):
            #     img3[x, y] = 255        # [0, 128, 128] - yellow - cup
            # elif pixel[2] == (int)(0) and pixel[1] == (int)(128) and pixel[0] == (int)(0):
            #     img4[x, y] = 255        # [0, 128, 0] - green - bowl
            # elif pixel[2] == (int)(0) and pixel[1] == (int)(0) and pixel[0] == (int)(128):
            #     img5[x, y] = 255        # [128, 0, 0] - blue - spoon
            # elif pixel[2] == (int)(128) and pixel[1] == (int)(0) and pixel[0] == (int)(128):
            #     img6[x, y] = 255        # [128, 0, 128] - purpue -oats
            # print(pixel)

    sub_masks[0] = img1  # hand
    # sub_masks[1] = img2  # milk
    # sub_masks[2] = img3  # cup
    # sub_masks[3] = img4  # bowl
    # sub_masks[4] = img5  # spoon
    # sub_masks[5] = img6  # oats
    return sub_masks

# dex-ycb hand segmentation function

def dexycb_hand_seg_func_mivos(num_samples=-1, 
                               ignore_background=False, 
                               data_dir='/home/kolz14w/haucode/hw2/assets/mask_rcnn/content', 
                               step_size=2):
    subjects = [f for f in os.listdir(data_dir) if 'subject' in f]
    print('Subjects: ',subjects)
    lst = []
    
    prefix = f"{data_dir}"
    length = num_samples
    print(length)
    for i in range(0, length, 1):
        # color_file = f"{prefix}/images/frame%04d.jpg" % (i * step_size)
        color_file = f"{prefix}/images/frame%04d.jpg" % (i * step_size)
        seg_mask = f"{prefix}/mask/%05d.png" % (i * step_size)
        print(color_file)
        print(seg_mask)
        

        assert os.path.exists(color_file)
        assert os.path.exists(seg_mask)
        seg_img = cv2.imread(seg_mask)
        height, width, channels = seg_img.shape

        a = create_sub_masks(seg_img, height, width)
        
        # print(a)
        # for i in a.keys():
        #     print(a[i].any())
        
        annotations = []

        for j in range(len(a)):
            
            rel_mask = coco_mask.encode(np.asfortranarray(a[j]))
            rows, cols = np.where(a[j])
            category_label, category_id = get_category2(j)
            print(rows.any())

            if rows.any():
                print("Entered Annotator")
                annotation = {
                    'bbox': [min(cols), min(rows), max(cols), max(rows)],
                    'bbox_mode': BoxMode.XYXY_ABS,
                    'category_id': category_id,
                    'segmentation': rel_mask
                }

                print(annotation)

                annotations.append(annotation)

        dct = {
            'file_name': color_file,
            'height': height,
            'width': width,
            'image_id': i,
            'annotations': annotations,
        }

        lst.append(dct)
        if len(lst) == length:
            return lst

# Get object class

def get_category2(id):
    
    k = OBJ_CLASSES[id]

    return k, INV_OBJ_CATEGORIES[k]

# Dataset registeration

def data_reg(dataset_name, dataset_path):
    DatasetCatalog.clear()
    MetadataCatalog.clear()
    DatasetCatalog.register(dataset_name, partial(dexycb_hand_seg_func_mivos,
                                                 num_samples = 20,
                                                 ignore_background=True,
                                                 data_dir = dataset_path,
                                                 step_size = 3))

class Detectron2Wrapper:
    
    def __init__(self, cfg_path = '/home/kolz14w/Proj_Master/CAOCM/CACOM/cfgCAOCM.yaml', 
                 dataset_path = '/home/kolz14w/Proj_Master/CAOCM/CACOM/dataset',
                 model_path="/home/kolz14w/Proj_Master/haucode/hw2/assets/mask_rcnn/pretrained/model_final.pth"):
        cfg = get_cfg()
        cfg.merge_from_file(cfg_path)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
        cfg.MODEL.WEIGHTS = model_path
        cfg.DATASETS.TEST = ('Mivos',)
        self.dataset_name = 'Mivos'
        self.dataset_path = dataset_path
        self.predictor = DefaultPredictor(cfg)
        self.gen_meta()
    
    def gen_meta(self):
        data_reg(self.dataset_name, self.dataset_path)
        dataset = DatasetCatalog.get('Mivos')
        self.meta = MetadataCatalog.get('Mivos')
        num_categories = len(OBJ_CLASSES)
        self.meta.thing_classes = [OBJ_CLASSES[i] for i in range(num_categories)]
        
    def process(self, img_gbr):
        outputs = self.predictor(img_gbr)
        mask = outputs['instances'].get('pred_masks').to('cpu')
        classes = outputs['instances'].get('pred_classes').to('cpu')
        mask = mask.numpy()
        classes = classes.numpy()
        return mask, classes

# Comparation with the Mivos output

def eval_mivos(input_image, original_mask, det):
    
    pred_mask, classes = det.process(input_image)
    w, h = original_mask.shape[:2]
    orig_mask = create_sub_masks(original_mask, w, h)
    
    orig_mask_array = np.zeros(pred_mask.shape)
    for key in orig_mask.keys():
       orig_mask_array[key, :, :] = orig_mask[key].astype('bool')
    
    # reordering the masks
    reordered_pred_mask = np.copy(pred_mask)
    reordered_pred_mask[0,:,:] = pred_mask[4, :, :]
    reordered_pred_mask[1,:,:] = pred_mask[0, :, :]
    reordered_pred_mask[2,:,:] = pred_mask[3, :, :]
    reordered_pred_mask[3,:,:] = pred_mask[2, :, :]
    reordered_pred_mask[4,:,:] = pred_mask[1, :, :]
    
    diff = np.absolute(reordered_pred_mask - orig_mask_array)
    
    # show difference visually
    for i in range(diff.shape[0]):
        plt.imshow(diff[i,:,:])
        plt.show()
        
    # return pred_mask, orig_mask_array
if 'cfg' in dir():
    print('Configuration has been initialized!')
    print("Skip configuration settings!")
else:
    print("Start initializing configuration for Detectron2")
    cfg = Init_Save_Cfg(# cfg_path="/home/kolz14w/haucode/hw2/cfghw2_2_jup.yaml",
                        model_path = "/home/kolz14w/Proj_Master/CAOCM/CACOM/output/model_final.pth",
                        num_categories=1)

# cfg = get_cfg()
# cfg.merge_from_file('cfghw2_2_jup.yaml')

# Set expected object classes
OBJ_CLASSES = {
    0: '00_Foot'
}

INV_OBJ_CATEGORIES = {v: k for k,v in OBJ_CLASSES.items()}
print(INV_OBJ_CATEGORIES)

def Model_training(cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.train()
    return trainer

# data_reg("Mivos", '/home/kolz14w/Proj_Master/CAOCM/CACOM/dataset')

# print('Training start.')

# Model_training(cfg)

if 'det' in locals().keys():
    print('Detectron2Wrapper is already initialized')
else:
    print('Initializing Detectron2Wrapper.')    
    det = Detectron2Wrapper(# cfg_path="/home/kolz14w/Proj_Master/haucode/hw2/cfghw2_2_jup.yaml",
                            model_path = "/home/kolz14w/Proj_Master/CAOCM/CACOM/output/model_final.pth"
                            )

# print('Initializing Visualizer.')

test_path="/home/kolz14w/Proj_Master/CAOCM/CACOM/dataset/images/frame0001.jpg"

# init_visualizer(det, test_path)



# ############################ compare the difference ###########################

# img_test = cv2.imread('mask_rcnn/Mivos_cus/images/frame0006.jpg')
# mask_test = cv2.imread('mask_rcnn/Mivos_cus/mask/00006.png')

# eval_mivos(img_test, mask_test, det)


############################## Mask Processing

# mask for color image generation

def color_mask_gen(input_img, det):
    mask, classes = det.process(input_img)
    color_mask = np.zeros(np.hstack([mask.shape[1:],[3]]))
    
    overall_mask = np.zeros(mask.shape[1:])
    for inst in range(mask.shape[0]):
        overall_mask = overall_mask + mask[inst,:,:]
    
    bool_mask = (overall_mask > 0)
    
    for layer in range(color_mask.shape[2]):
        color_mask[:,:,layer] = bool_mask
    
    return color_mask

def obj_segmentation(img, mask):
    seg_img = (img * mask).astype('float32') / 255.0
    gray_img = cv2.cvtColor(seg_img * 255.0, cv2.COLOR_BGR2GRAY).astype('uint8')
    return seg_img, gray_img
    
test_img = cv2.imread(test_path)

color_mask = color_mask_gen(test_img, det).astype('float32')
seg_img, gray_img = obj_segmentation(test_img, color_mask)

plt.imshow(seg_img)

# find best parameters for blob
def best_fit_keypoints(img, max_minArea, num_points):
    keypoints_dict = []
    keypoints_list = []
    minArea_list = []
    for i in range(1,max_minArea):
        keypoints = dotdetector(img,i)
        if len(keypoints) != 0:
            keypoints_list.append(keypoints)
            minArea_list.append(i)
    keypoints_dict = dict(zip(minArea_list, keypoints_list))
    num_keypoints = []
    for i in keypoints_dict.keys():
        num_keypoints_cur = len(keypoints_dict[i])
        num_keypoints.append(num_keypoints_cur)

    dist = abs(np.array(num_keypoints) - np.array([num_points] * len(num_keypoints)))
    index = np.argmin(dist)
    keypoints_with_best_fit_minArea = keypoints_dict[list(keypoints_dict.keys())[index]]
    return keypoints_with_best_fit_minArea

# define dotdetector at first to detect blob features
def dotdetector(img,minArea):
    img_int8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_int8 = np.copy(img_int8).astype("uint8")
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True   #setting up Blob detector
    params.minArea = minArea
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img_int8)
    return keypoints

# draw keypoints on original image
def visualizer(img, keypoints):
    blobs = cv2.drawKeypoints(img, keypoints, np.zeros((1,1)), (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.drawContours(test_img,seg_img, -1, (0,255,0), 3)

    cv2.imshow('blobs',blobs)
    key = cv2.waitKey(-1)
    if key == 27:
        cv2.destroyAllWindows()

# draw point distribution on foot
def find_marker(keypoints, visualize = True):
    blobs = cv2.drawKeypoints(np.zeros(test_img.shape).astype('uint8'), keypoints, np.zeros((1,1)), (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    blobsseg, blobsgray = obj_segmentation(blobs, color_mask)
    if visualize:        
        cv2.imshow('seg', blobsseg)
    return blobsseg


keypoints = best_fit_keypoints(test_img, 25, 30)
# visualizer(test_img, keypoints)
keypoints_on_foot = find_marker(keypoints, 1)    











