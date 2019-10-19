# -*- coding: utf-8 -*-
"""
Konstantin Schuckmann
Get used to OpenCV
19.10.19 
"""
import numpy as np
import cv2

# Read image from file 
img_color = cv2.imread("/Users/Kostja/Desktop/Master/Sem 5 (19:20 WiSe)/Learning from Images/Assignments/lfi-01/Lenna.png",1)

img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
img_gray = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2RGB)

img = np.concatenate((img_gray,img_color), axis = 1)

cv2.imshow('image', img)


