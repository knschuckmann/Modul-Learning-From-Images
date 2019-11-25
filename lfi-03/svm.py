#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 18:08:29 2019

@author: Kostja
"""

import numpy as np
import cv2
import glob
from sklearn import svm
import matplotlib.pyplot as plt

import os
from itertools import groupby 

############################################################
#
#              Support Vector Machine
#              Image Classification
#
############################################################

def load_image(path):
    """Input: path to image
    return: image
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def find_index_Img(paths, startDiffer):
    """Input: paths of inages and the starting point from where the strings schould differ
    Return a list with points where successive images differ
    """
    group = []
    length = len(paths)
    for i in range(length - 1):
        if imgPaths[i][startDiffer:startDiffer + 2] != imgPaths[i + 1][startDiffer:startDiffer + 2]:
            group += [i+1]
    return group

def plot_imgs_with_subplots(rows,cols, group, imgs, axs):
    """Input 
    Return 
    """
    imgNr = 0
    for i in range(rows):
        for j in range(cols):
            if imgNr in group or imgNr == len(imgs):
                imgNr += 1
                break
            axs[i,j].imshow(imgs[imgNr])
            imgNr += 1        
    pass
    
# Set the absolute path to work on the right folder
os.chdir("/Users/Kostja/Desktop/Master/Sem 5 (19:20 WiSe)/Learning from Images/Assignments/lfi-03/")


# 1. Implement a SIFT feature extraction for a set of training images ./images/db/train/** (see 2.3 image retrieval)
# use 256x256 keypoints on each image with subwindow of 15x15px

imgPaths = glob.glob("./images/db/train/*/*")
groups = find_index_Img(imgPaths, 18)
images = [load_image(imgPaths[i]) for i in range(len(imgPaths))]    
fig, axs =  plt.subplots(nrows=3, ncols=10,figsize=(13, 4),subplot_kw={'xticks': [], 'yticks': []})
plot_imgs_with_subplots(3,10,groups, images, axs)

axs[0,9].imshow(images[1])


# 2. each descriptor (set of features) need to be flattened in one vector
# That means you need a X_train matrix containing a shape of (num_train_images, num_keypoints*num_entry_per_keypoint)
# num_entry_per_keypoint = histogram orientations as talked about in class
# You also need a y_train vector containing the labels encoded as integers


# 3. We use scikit-learn to train a SVM classifier - however you need to test with different kernel options to get
# good results for our dataset.


# 4. We test on a variety of test images ./images/db/test/ by extracting an image descriptor
# the same way we did for the training (except for a single image now) and use .predict()
# to classify the image

# 5. output the class + corresponding name
