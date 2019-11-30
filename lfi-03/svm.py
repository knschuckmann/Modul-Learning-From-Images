#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 18:08:29 2019
This Script Trains a SVM model to predict classes. It works on Images among with created Labels that are loaded 
and trains a model on a self initialized kernel. At the very end there is a test set that is predicted by the given model.

Information and Sources:
    * https://scikit-learn.org/stable/modules/svm.html#svm-classification
    
    
@author: Konstantin Schuckmann
"""

import numpy as np
import cv2
import glob
from sklearn import svm
import matplotlib.pyplot as plt

import os

############################################################
#
#              Support Vector Machine
#              Image Classification
#
############################################################

def load_image(path):
    """Input: path to image
    return: image in RGB
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
        if paths[i][startDiffer:startDiffer + 2] != paths[i + 1][startDiffer:startDiffer + 2]:
            group += [i+1]
    return group

def plot_imgs_with_subplots(rows,cols, group, imgs, axs):
    """Input  rows and columsn of subplots among with an array consisting of numbers of different grouped images, the images themself and
    axes of subplots to plot the images in. 
    Passes a plot of all images devided into the given groups
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

def create_keypoints(w, h, gsize=1, keypointSize=11):
    """Input wide and high, the size between the keypoints and the keypointsize itself
    Return creatded Keypoints for SIFT algorithm
    """
    # stepsize is gsize
    col = np.arange(0,w,gsize)
    row = np.arange(0,h,gsize)
    
    keypoints = [cv2.KeyPoint(x,y,keypointSize) for x in col for y in row]
    
    return keypoints

def labelEnc(label, imagePath):
    """Input label and imagepath
    check if the imagePath contains the label 
    Return the enumeration of the label 
    """
    for index, i in enumerate(label):
        if i in imagePath:
            return index

    
# Set the absolute path to work on the right folder
os.chdir("/Users/Kostja/Desktop/Master/Sem 5 (19:20 WiSe)/Learning from Images/Assignments/lfi-03/")

imgPathsTrain = glob.glob("./images/db/train/*/*")
groups = find_index_Img(imgPathsTrain, 18)
images = [load_image(imgPathsTrain[i]) for i in range(len(imgPathsTrain))]    
fig, axs =  plt.subplots(nrows=3, ncols=10,figsize=(13, 4),subplot_kw={'xticks': [], 'yticks': []})
plot_imgs_with_subplots(3,10,groups, images, axs)


# 1. Implement a SIFT feature extraction for a set of training images ./images/db/train/** (see 2.3 image retrieval)
# use 256x256 keypoints on each image with subwindow of 15x15px

sift = cv2.xfeatures2d.SIFT_create()
high, weigh = images[0].shape[0:2]
keyPoi = create_keypoints(weigh,high, keypointSize=15)

# 2. each descriptor (set of features) need to be flattened in one vector
# That means you need a X_train matrix containing a shape of (num_train_images, num_keypoints*num_entry_per_keypoint)
# num_entry_per_keypoint = histogram orientations as talked about in class
# You also need a y_train vector containing the labels encoded as integers
descriptors = []
numImages = len(images)
descriptors = [sift.compute(images[i], keyPoi) for i in range(numImages)]

descFlat = [descriptors[i][1].flatten() for i in range(numImages)]
X_train = np.array(descFlat)

labels = ['flower','car','face']
y_train = np.array([labelEnc(labels,path) for path in imgPathsTrain])

# 3. We use scikit-learn to train a SVM classifier - however you need to test with different kernel options to get
# good results for our dataset.
kernels = ['linear', 'poly'] # 'rbf', 'sigmoid'] # not so good
clf = svm.SVC(kernel=kernels[0]) 
clf.fit(X_train,y_train)

# 4. We test on a variety of test images ./images/db/test/ by extracting an image descriptor
# the same way we did for the training (except for a single image now) and use .predict()
# to classify the image
imgPathsTest = glob.glob("./images/db/test/*")

for index, img in enumerate(imgPathsTest):
    imgTest = load_image(img)
    desc = sift.compute(imgTest, keyPoi)
    desc = desc[1].flatten()
    prediction = clf.predict(desc.reshape(1,-1))[0]
    imgname = img[img.rfind('/')+1:]
    print(f'Image: {imgname}\tPrediction: {labels[prediction]}')
















