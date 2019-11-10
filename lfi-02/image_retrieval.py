#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 17:46:10 2019

@author: Konstantin Schuckmann
"""
import cv2
# The glob module finds all the pathnames matching a specified 
# pattern according to the rules used by the Unix shell, although 
# results are returned in arbitrary order
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


# implement distance function
def distance(a, b):
    """Input two pixels 
    calculate the euclidean distance
    Return: Euclidean distance
    """
    return np.linalg.norm(a-b)


def load_image(path):
    img = cv2.imread(path, cv2.COLOR_BGR2GRAY)
    return img

def create_keypoints_Uniform(w, h, keypointSize):
    """Input width, hight of image and the gridsize
    The function calculates a Grid for the size of the image and 
    Return a list [x,y,diameter]
    
    """
    keypoints = []
    
    x_dir = np.arange(0,w,keypointSize)
    y_dir = np.arange(0,h,keypointSize)
    keypoints = np.meshgrid(x_dir,y_dir)

    keypoints = [[keypoints[0][i][j],keypoints[1][i][j]] for i in range(len(x_dir)) for j in range(len(x_dir))]
    keypoints = np.reshape(keypoints,(len(keypoints),2))
    return keypoints


os.chdir("/Users/Kostja/Desktop/Master/Sem 5 (19:20 WiSe)/Learning from Images/Assignments/lfi-02")

# 1. preprocessing and load
img = glob.glob('./images/db/**/*.jpg', recursive = True)
images = [load_image(img[i]) for i in range(len(img))]

# 2. create keypoints on a regular grid (cv2.KeyPoint(r, c, keypointSize), as keypoint size use e.g. 11)
keyPointSize = 11
keypoints = create_keypoints_Uniform(256, 256,keyPointSize)

# create the keys
keys = [cv2.KeyPoint(keypoints[i][0], keypoints[i][1], keyPointSize) for i in range(len(keypoints))]

sift = cv2.xfeatures2d.SIFT_create()

# 3. use the keypoints for each image and compute SIFT descriptors
#    for each keypoint. this compute one descriptor for each image.
# descriptors is devided in keypoints and descriptors like mentioned in 
descriptors = []
# 16 x 16 neighbourhood arround keypoints each devided into 16 subblox of the size of 4x4
# each of this subblocks has 8 bin orientations
# result is for each keypoint a 128 ( = 16sobbloxks * 8 bins)  bin values
descriptors = [sift.compute(images[i], keys) for i in range(len(img))]

# 4. use one of the query input image to query the 'image database' that
#    now compress to a single area. Therefore extract the descriptor and
#    compare the descriptor to each image in the database using the L2-norm
#    and save the result into a priority queue (q = PriorityQueue())
q = PriorityQueue()
#[q.put((distance(descriptors[test][1],descriptors[train][1]), str((test,train)))) for test in range(4) for train in range(4,len(descriptors))]
# 5. output (save and/or display) the query results in the order of smallest distance
for test in range(4):
    for train in range(4,len(descriptors)):
        q.put((distance(descriptors[test][1],descriptors[train][1]),[img[test],img[train]]))

while not q.empty():        
    print(q.get()[0])
    

for i in range(4):

img = load_image(q.queue[0][1][0])
img2 = []
for j in range(len(q.queue)):
    img2 = load_image(q.queue[j][1][1])
    

    
cv2.imshow('as',)np.concatenate((,),axis = 1)

nr = np.arange(80)

fig, axs = plt.subplots(nrows=4, ncols=20, figsize=(9, 6),
                        subplot_kw={'xticks': [], 'yticks': []})


for ax, nr1 in zip(axs.flat, nr):
    ax.imshow(load_image(q.queue[nr1][1][1]))

plt.tight_layout()
plt.show()

i1 = load_image(q.get()[1][0])
i2 = load_image(q.get()[1][1])
i = np.concatenate((i1,i2), axis = 1)
cv2.imshow('seds',i)

plt.figure()
