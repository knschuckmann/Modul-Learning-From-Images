#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 00:56:35 2019

@author: Kostja
"""

import numpy as np
import cv2
import math
import sys



def initialize(img, clustNum):
    """inittialize the current_cluster_centers array for each cluster with a random pixel position"""
    randomNumb = np.random.randint(0,512,(clustNum,2))
    current_cluster_centers = img[randomNumb[:,0],randomNumb[:,1]]
    
    clusMask = np.random.randint(0,clustNum,(img.shape[0],img.shape[0],1))
    
    return [current_cluster_centers, clusMask]

# implement distance metric - e.g. squared distances between pixels
# a and b should be pixels on their own 
def distancePixelwise(pix_1, pix_2):
    temp = 0
    for num in range(0,pix_1.shape[0]):
        temp = temp + (pix_1[num] - pix_2[num])**2
        temp = np.sqrt(temp)
    return temp
    
def assign(img, resultImg, current_clusters, clusMask, clustNum):
    
    distMat = np.zeros((imgraw.shape[0],img.shape[1],clustNum))
    concatMat = np.zeros((img.shape[0],img.shape[1],img.shape[2]+3))
    for col in range(0, img.shape[0]):
        for row in range(0,img.shape[1]):
            resultImg[col,row] = current_clusters[clusMask[col][row]]
            for cluster in range(0,clustNum):
                if clusMask[col][row] == cluster:
                    # hier wird oft nur noch auf 2 cluster aufgeteilt so das das dritte verschwindet
                    distMat[col,row,cluster] = distancePixelwise(img[col,row],resultImg[col,row])     
            clusMask[col,row] = distMat[col,row].tolist().index(distMat[col,row].min())
            
            concatMat[col,row] = np.concatenate([resultImg[col,row], clusMask[col,row],[col],[row]])
            
    return concatMat

def update(concatMat, current_clusters, clustNum):
    
    for cluster in range(0,clustNum):
        for rgb in range(0,current_clusters.shape[1]):
            current_clusters[cluster,rgb] = np.mean(concatMat[concatMat[concatMat[:,:,3] == cluster,4].astype(int),concatMat[concatMat[:,:,3] == cluster,5].astype(int)][:,rgb])    
                  
    clusMask = concatMat[:,:,3]
    
    return [clusMask, current_clusters]


# load image
imgraw = cv2.imread('/Users/Kostja/Desktop/Master/Sem 5 (19:20 WiSe)/Learning from Images/Assignments/lfi-01/images/Lenna.png')

# num of cluster
numclusters = 3

# set seed for reproducability
np.random.seed(200)     
current_cluster_centers, clustermask = initialize(imgraw,numclusters)

result = np.zeros(imgraw.shape, np.uint8)

res = assign(imgraw,result,current_cluster_centers,clustermask, numclusters)
clustermask,current_cluster_centers = update(res, current_cluster_centers,numclusters)

cv2.imshow('ner', result)



