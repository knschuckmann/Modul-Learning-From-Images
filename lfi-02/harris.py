#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 12:40:23 2019
This is a implementation of a harris corner detector. This skript takes the gradients of an image
and calculates the harris corners by calculating the eigenvalues with gradient Matrix.

Sources and Information:
    * entries of the matrix M = \sum_{3x3} [ G_xx Gxy; Gxy Gyy ]
    * no need to calc eigenvalues because 
        det_M = Lamb_1*Lamb_2
        trace_M = Lamb_1 + Lamb_2


@author: Konstantin Schuckmann
"""

import cv2
import numpy as np
# important for defining own path 
import os

# os.chdir("/Users/Kostja/Desktop/Master/Sem 5 (19:20 WiSe)/Learning from Images/Assignments/lfi-02")
# Load image and convert to gray and floating point
img = cv2.imread('./images/Lenna.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

ksize = 3

# Define sobel filter and use cv2.filter2D to filter the grayscale image
G_x = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=ksize)
G_y = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=ksize)

# Compute G_xx, G_yy, G_xy and sum over all G_xx etc. 3x3 neighbors to compute
G_xx = G_x * G_x
G_yy = G_y * G_y
G_xy = G_x * G_y

# Note1: this results again in 3 images sumGxx, sumGyy, sumGxy
# Hint: to sum the neighbor values you can again use cv2.filter2D to do this efficiently
kernel = np.ones((ksize,ksize), dtype = np.uint8)

# ddepth = - 1 same depth as source 
sumG_xx = cv2.filter2D(G_xx,ddepth = -1 ,  kernel = kernel)
sumG_yy = cv2.filter2D(G_yy,ddepth = -1 ,  kernel = kernel)
sumG_xy = cv2.filter2D(G_xy,ddepth = -1 ,  kernel = kernel)

# Define parameter
k = 0.04
threshold = 0.01

# Compute the determinat and trace of M using sumGxx, sumGyy, sumGxy. With det(M) and trace(M)
det_M = sumG_xx * sumG_yy - np.power(sumG_xy,2)
trace_M = sumG_xx + sumG_yy

# compute the resulting image containing the harris corner responses

harris = det_M - k * np.power(trace_M,2)

# Filter the harris 'image' with 'harris > threshold*harris.max()'
# this will give you the indices where values are above the threshold.
# These are the corner pixel you want to use
harris_thres = np.zeros(harris.shape, np.uint8)
harris_thres[harris > threshold*harris.max()] = [255]

# The OpenCV implementation looks like this - please do not change
harris_cv = cv2.cornerHarris(gray,3,3,k)

# intialize in black - set pixels with corners in with
harris_cv_thres = np.zeros(harris_cv.shape)
harris_cv_thres[harris_cv>threshold*harris_cv.max()]=[255]

# just for debugging to create such an image as seen
# in the assignment figure.
img[harris>threshold*harris.max()]=[255,0,0]

# please leave this - adjust variable name if desired
print("====================================")
print("DIFF:", np.sum(np.absolute(harris_thres - harris_cv_thres)))
print("====================================")

cv2.imwrite("Harris_own.png", harris_thres)
cv2.imwrite("Harris_cv.png", harris_cv_thres)
cv2.imwrite("Image_with_Harris.png", img)

cv2.namedWindow('Interactive Systems: Harris Corner')

while True:
    ch = cv2.waitKey(0)
    if ch == 27:
        cv2.destroyAllWindows()
        break
    cv2.imshow('harris',harris_thres)
    cv2.imshow('harris_cv',harris_cv_thres)
