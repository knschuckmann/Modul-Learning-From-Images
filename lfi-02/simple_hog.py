#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 19:24:31 2019

@author: Konstantin Schuckmann
"""
import numpy as np
import cv2
import math
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import glob
import os

###############################################################
#
# Write your own descriptor / Histogram of Oriented Gradients
#
###############################################################


def plot_histogram(hist, bins,path):
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.figure()
    plt.bar(center, hist, align='center', width=width)
    plt.title(path)
    plt.show()


def compute_simple_hog(imgcolor, keypoints, imgpath):

    # convert color to gray image and extract feature in gray

    # compute x and y gradients (sobel kernel size 5)
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html
    # This describes a better way to use sobel filter eventhough the img is in uint8
    # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U did not work for angle calculation
    sobel_x = cv2.Sobel(imgcolor,cv2.CV_64F,1,0,ksize=5)
    sobel_y = cv2.Sobel(imgcolor,cv2.CV_64F,0,1,ksize=5)

    # compute magnitude and angle of the gradients
    # The magnitude represents the intensity of the pixel and the orientation gives the direction for the same.
    # https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/
    angleGr = cv2.phase(sobel_x,sobel_y, angleInDegrees = False)
    magGr = cv2.magnitude(sobel_x,sobel_y)

    # go through all keypoints and compute feature vector
    descr = np.zeros((len(keypoints), 8), np.float32)
    count = 0
    for kp in keypoints:
        # print kp.pt, kp.size
        print(f'Keypoint: {kp.pt}\nSize: {kp.size}')
        x, y = kp.pt            
        # extract gradient magnitude in keypoint subwindow
        
        pt = calc_surrounding_keypoint(x,y,kp.size)
        magKp = magGr[pt[0][0]:pt[0][1],pt[1][0]:pt[1][1]]
        # extract angle in keypoint sub window
        angleKp = angleGr[pt[0][0]:pt[0][1],pt[1][0]:pt[1][1]]
        
        # create histogram of angle in subwindow BUT only where magnitude of gradients is non zero! Why? Find an
        # answer to that question use np.histogram
        
        angleKp = angleKp[magKp > 0]
        # Where magnitude is Zero this regions are not interesting. They tend to be without Edges or anything else
        # this regions are mostly plain like a white sheet of paper
        #(hist, bins) = np.histogram(...)
        
        (hist, bins) = np.histogram(angleKp, bins = 8, density = True, range = (0, 2*np.pi))
        plot_histogram(hist, bins, imgpath)

        descr[count] = hist

    return descr

def load_image(path):
    img = cv2.imread(path, cv2.COLOR_BGR2GRAY)
    img = np.float32(img)
    return img

def calc_surrounding_keypoint(point_x_of_kp, point_y_of_kp, size):
    mod = lambda x,md: x%md      
    # not taken into account if the picture has the calculated values
    dev = math.floor(size/2)
    
    pt_x_top_left = point_x_of_kp - dev
    pt_y_top_left = point_y_of_kp - dev
    
    pt_x_botom_right = point_x_of_kp + dev
    pt_y_botom_right = point_y_of_kp + dev 
    
    if mod(size,2) == 1:
        pt_x_botom_right = point_x_of_kp + dev + 1
        pt_y_botom_right = point_y_of_kp + dev + 1    
    
    return [np.uint8([pt_x_top_left,pt_x_botom_right]),np.uint8([pt_y_top_left,pt_y_botom_right])]

keypoints = [cv2.KeyPoint(15, 15, 11)]

# test for all test images
# preprocessing and load
os.chdir("/Users/Kostja/Desktop/Master/Sem 5 (19:20 WiSe)/Learning from Images/Assignments/lfi-02")
img = glob.glob('./images/hog_test/*.jpg')
images = [load_image(img[i]) for i in range(len(img))]


descriptor = []
# test = cv2.imread('./images/hog_test/diag.jpg')
descriptor = [compute_simple_hog(images[i], keypoints, img[i]) for i in range(len(images))]




