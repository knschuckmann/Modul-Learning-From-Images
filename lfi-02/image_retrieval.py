#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 17:46:10 2019
This function calculates SIFT descriptors for images using uniform distributed keypoints.
It also ques the results so one can have a better glance on similar pictures using the given Database.

Information and Sources:
    * The glob module finds all the pathnames matching a specified 
        pattern according to the rules used by the Unix shell, although 
        results are returned in arbitrary order
    
    
@author: Konstantin Schuckmann
"""
import cv2
import glob
import numpy as np
import os
from queue import PriorityQueue
import matplotlib.pyplot as plt

############################################################
#
#              Simple Image Retrieval
#
############################################################

def distance(a, b):
    """Input two pixels, or two images 
    calculate the euclidean distance
    Return: Euclidean distance
    """
    return np.linalg.norm(a-b)

def load_image(path):
    """Input: path to image
    return: image
    """
    img = cv2.imread(path,cv2.COLOR_BGR2GRAY)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def create_keypoints(w, h,gsize=11,keypointSize=11):    
    col = np.arange(0,w,gsize)
    row = np.arange(0,h,gsize)
    
    keypoints = [cv2.KeyPoint(x,y,keypointSize) for x in col for y in row]
    
    return keypoints


def create_keypoints_Uniform(w, h, keypointSize):
    """Input width, hight of image and the gridsize
    The function calculates a Grid for the size of the image and 
    Return a list [x,y,diameter]
    
    """
    keypoints = []
    
    x_dir = np.arange(0,w,keypointSize)
    y_dir = np.arange(0,h,keypointSize)
    keypoints = np.meshgrid(x_dir,y_dir)
    
    # go through each x, and y point and connect them 
    keypoints = [[keypoints[0][i][j],keypoints[1][i][j]] for i in range(len(x_dir)) for j in range(len(x_dir))]
    # Reshape the result for further use in cv2.Keypoints
    keypoints = np.reshape(keypoints,(len(keypoints),2))
    
    return keypoints

# only used on my machine for setting the direction
os.chdir("/Users/Kostja/Desktop/Master/Sem 5 (19:20 WiSe)/Learning from Images/Assignments/lfi-02")

# 1. preprocessing and load
train_img = glob.glob('./images/db/train/*/*.jpg')
test_img = glob.glob('./images/db/test/*.jpg')
images_train = [load_image(train_img[i]) for i in range(len(train_img))]
images_test = [load_image(test_img[i]) for i in range(len(test_img))]

# 2. create keypoints on a regular grid (cv2.KeyPoint(r, c, keypointSize), as keypoint size use e.g. 11)
keyPointSize = 11
keypoints = create_keypoints_Uniform(256, 256,keyPointSize)

# create the keys
keys = [cv2.KeyPoint(keypoints[i][0], keypoints[i][1], keyPointSize) for i in range(len(keypoints))]

# 3. use the keypoints for each image and compute SIFT descriptors
#    for each keypoint. this calculation computes one descriptor for each image.
#    descriptors are devided in keypoints and descriptors
sift = cv2.xfeatures2d.SIFT_create()
descriptors_test = []
descriptors_train = []
# 16 x 16 neighbourhood arround keypoints, each devided into 16 subblocks of the size of 4x4 pixels
# each of this subblocks has 8 bin orientations
# result in 128 ( = 16sobbloxks * 8 bins) bin values for each keypoint 
descriptors_train = [sift.compute(images_train[i], keys) for i in range(len(train_img))]
descriptors_test = [sift.compute(images_test[i], keys) for i in range(len(test_img))]

# 4. use one of the query input image to query the 'image database' that
#    now compress to a single area. Therefore extract the descriptor and
#    compare the descriptor to each image in the database using the L2-norm
#    and save the result into a priority queue (q = PriorityQueue())
q = PriorityQueue()

# 5. output (save and/or display) the query results in the order of smallest distance
# 4 because there are 4 test images so range does 0, 1, 2, 3
for test in range(len(test_img)):
    print(test_img[test])
    # for loop for train images
    for train in range(len(test_img),len(descriptors_train)):
        q.put((distance(descriptors_test[test][1],descriptors_train[train][1]),train_img[train]))

# can be used but sure yet if it is right but it displays the que from small to big
#while not q.empty():        
#   print(q.get()[0])
#   print(q.get()[1])
#      
########################################################
# here follows a possibility to display the images
# but it seems not to be right
# possible reasons make first picture apper on left followed by 4 times 2x10 picture like
# in figure of exercise

nr = np.arange(80)

fig, axs = plt.subplots(nrows=4, ncols=20, figsize=(9, 6),subplot_kw={'xticks': [], 'yticks': []})

for ax, nr1 in zip(axs.flat, nr):
    ax.imshow(load_image(q.queue[nr1][1]))
