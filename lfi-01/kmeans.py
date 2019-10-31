#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 00:56:35 2019
Learning from Images first Assignment 
implementing the k-means algorithm for learning purposes
@author: Konstantin Schuckmann
"""

import numpy as np
import cv2
import sys

def initialize(img, clustNum):
    """given the image and the number of clusters k
    inittialize the current_cluster_centers array for each cluster with a random pixel position
    and a clustermask
    return: currents cluster centers as rgb and a cluster mask in the shape of image but only 1 dim in z direction
    """
    # generate random numbers 
    randomNumb = np.random.randint(0,img.shape[0],(clustNum,2))
    # use generatet numbers to choose from image
    current_cluster_centers = img[randomNumb[:,0],randomNumb[:,1]].astype(float)
    clusMask = np.random.randint(0,clustNum,(img.shape[0],img.shape[0],1))
    
    return [current_cluster_centers, clusMask]


def distancePixelwise(pix_1, pix_2):
    """ given two pixels of same length
    calculate the euclidean distance 
    return the euclidean distance
    """
    n = pix_1.shape[0]
    temp = [(pix_1[num] - pix_2[num])**2 for num in range(n)]
    temp = np.sqrt(sum(temp))

    return temp
    
def assign(img, resultImg, current_clusters, clusMask, clustNum):
    """ given the image, a resultmatrix in image shape, the current cluster means, the cluster mask and the number of clusters
    Assign function calculates the distances between the means and the img pixels and provides the nearest cluster throughout the pixels
    return a concatenate matrix (img,clustermask,index col from img,index row from img) and an overall distance of whole img to clusters means
    """
    
    hight, width, dim = img.shape
    
    distMat = np.zeros((hight,width,clustNum))
    concatMat = np.zeros((hight,width,dim + 3))
    overall_dist = np.zeros(clustNum)
    
    # go through the whole img
    for col in range(hight):
        for row in range(width):
            # important even though it is not returned but still calculated outside the function
            resultImg[col,row] = current_clusters[clusMask[col][row]]
            # go through clusters to calc distance and the overall distance 
            for cluster in range(clustNum):
                # hier wird oft nur noch auf 2 cluster aufgeteilt so das das dritte verschwindet
                distMat[col,row,cluster] = distancePixelwise(img[col,row], current_clusters[cluster] )     
                overall_dist[cluster] = overall_dist[cluster] + distMat[col,row,cluster] 
            # get clustermask out of smalest distance index
            clusMask[col,row] = np.argmin(distMat[col,row])
            concatMat[col,row] = np.concatenate([img[col,row], clusMask[col,row],[col],[row]])
            
    return [concatMat,sum(overall_dist)]

def update(concatMat, current_clusters, clustNum):
    """ given the concatenate Matrix containing (img, cluster means and number of clusters
    update function updates the current cluster means 
    return updatet clustermask and new cluster means
    """
    
    for cluster in range(clustNum):
        for rgb in range(current_clusters.shape[1]):
            current_clusters[cluster,rgb] = np.mean(concatMat[concatMat[concatMat[:,:,3] == cluster,4].astype(int),concatMat[concatMat[:,:,3] == cluster,5].astype(int)][:,rgb])    
    
    # important for kmeans algorithm so clustermask wil change over the itterations
    clusMask = concatMat[:,:,3].astype(int)
    # reshape to old shape otherwise errors
    clusMask = clusMask.reshape(concatMat[:,:,3].shape[0],concatMat[:,:,3].shape[1],1)
    
    return [clusMask, current_clusters]

# https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
def mse(imageA, imageB):
    """ given two images of same shape 
    calculate the mean square error for comparrison
    return the mean squared error 
    """
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float")-imageB.astype("float"))**2)
    err = err/float(imageA.shape[0] * imageA.shape[1])
    
    return err


def kmeans(img,clustNum):
    """given an image
    Main k-means function iterating over max_iterations and stopping if
    the error rate of change is less then 2% for consecutive iterations, i.e. the
    algorithm converges. In our case the overall error might go up and down a little
    since there is no guarantee we find a global minimum.
    return a result matrix containing the best clustermeans to describe picture
    """
    max_iter = 10
    max_change_rate = 0.02
    dist = sys.float_info.max
    i = 0
    # set seed for reproducability
    # np.random.seed(23)     
    current_cluster_centers, clustermask = initialize(img,clustNum)
    result = np.zeros(img.shape, np.uint8)
    
    
    # loop over max iteration
    while(i <= max_iter):
        dist_old = dist
        
        res, dist = assign(img,result,current_cluster_centers,clustermask, clustNum)    
        clustermask,current_cluster_centers = update(res, current_cluster_centers,clustNum)
        
        print("Iteration: %i\tError: %.3f" %(i,dist))
        # how relatively big is the change 
        if  abs(dist_old-dist)/dist_old <= max_change_rate:
            break # if change detected loop should stop and return the result matrix
        i = i + 1
        
      #* calculate total error and print it - is missing  
    print('the MSE of result and starting image is: ', mse(img, result))
    return result

# load image
imgraw = cv2.imread('images/Lenna.png')

# num of cluster
numclusters = 3

# possibility to scale, i dont see why one should do so 
# scaling_factor = 0.5
# imgraw = cv2.resize(imgraw, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

# changing the colorspace to see the changes better 
# image = cv2.cvtColor(imgraw, cv2.COLOR_BGR2LAB)
image = cv2.cvtColor(imgraw,cv2.COLOR_BGR2HSV)
# image = imgraw
# cv2.imshow('ner', imgraw)
result = kmeans(image, numclusters)


h1, w1 = result.shape[:2]
h2, w2 = image.shape[:2]
vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
vis[:h1, :w1] = result
vis[:h2, w1:w1 + w2] = image

cv2.imshow("Color-based Segmentation Kmeans-Clustering", vis)