#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 12:23:20 2019

@author: denis
"""
"""
Kostja Comment: 
    Isnt the picture loaded in BGR instead of RGB?
    Somehow it works.. :-)
"""
import numpy as np
import cv2

# Load an color image in grayscale
img = cv2.imread('Lenna.png')

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)    #Convert RGB to Gray
#cv2.imshow('I',gray)

gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)   #Duplicate dimensions (m x n x c , where c = 3)
both = np.concatenate((gray,img),axis=1)         #concatenate both images

cv2.imshow('Image',both)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""
References:
    -https://opencv-python-tutroals.readthedocs.io/en/latest/
    -https://docs.scipy.org/doc/numpy/reference/
"""

