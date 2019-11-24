#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 19:24:31 2019
This skript displays a descriptor as a Histogram of Oriented 
Gradients (HOG), to detect shapes or edges. By computing the magnitude 
and the angle of a grayscaled image.

Additional information and sources:
    * https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html
        This describes a better way to use sobel filter eventhough the img is in uint8
        Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U did not work for angle calculation
    
    * The magnitude represents the intensity of the pixel and the orientation gives the direction for the same.
        https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/
  
    * Where magnitude is Zero this regions are not interesting. They tend to be without Edges or anything else
        this regions are mostly plain like a white sheet of paper
        

@author: Konstantin Schuckmann
"""
# no used because plt did not work with TkAgg
# import matplotlib
# matplotlib.use('TkAgg')

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import glob
# important for defining own path 
import os

###############################################################
#
# Write your own descriptor / Histogram of Oriented Gradients
#
###############################################################


def plot_histogram(hist, bins, path):
    """ Input: histogram values of image, number of bins for seperation, path or String for title
    This function plots a figure
    No return
    """
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.figure()
    plt.bar(center, hist, align='center', width=width)
    plt.title(path)
    plt.show()


def compute_simple_hog(imggray, keypoints, imgpath):
    """Input: image in gray, the calculated keypoints, imgpath for name of plot
    This function calculates the HOG by using the gradient the angle and the magnitude of 
    an Image for every given Keypoint. After that it plots the HOG.
    
    return: a descriptor
    
    """
    # compute x and y gradients (sobel kernel size 5)
    sobel_x = cv2.Sobel(imggray,cv2.CV_32FC1,1,0,ksize=5)
    sobel_y = cv2.Sobel(imggray,cv2.CV_32FC1,0,1,ksize=5)

    # compute magnitude and angle of the gradients
    angleGr = cv2.phase(sobel_x,sobel_y, angleInDegrees = False)
    magGr = cv2.magnitude(sobel_x,sobel_y)

    # go through all keypoints and compute feature vector
    descr = np.zeros((len(keypoints), 8), np.float32)
    count = 0
    for kp in keypoints:
        
        # print kp.pt, kp.size
        print(f'Keypoint: {kp.pt}\nSize: {kp.size}')
        x, y = kp.pt            
        
        pt = calc_surrounding_keypoint(x,y,kp.size)        
        # print the corners that define the subwindow
        print(f'The top left corner is {[pt[0][0],pt[1][0]]} and the buttom right corner is {[pt[0][1],pt[1][1]]}\n')

        # extract gradient magnitude in keypoint subwindow        
        magKp = magGr[pt[0][0]:pt[0][1],pt[1][0]:pt[1][1]]
        # extract angle in keypoint sub window
        angleKp = angleGr[pt[0][0]:pt[0][1],pt[1][0]:pt[1][1]]
        
        # create histogram of angle in subwindow BUT only where magnitude of gradients is non zero! Why? Find an
        # answer to that question use np.histogram
        angleKp = angleKp[magKp > 0]

        #(hist, bins) = np.histogram(...)
        (hist, bins) = np.histogram(angleKp, bins = 8, density = True, range = (0, 2*np.pi))
        plot_histogram(hist, bins, imgpath)

        descr[count] = hist

    return descr

def load_image_and_cvt(path):
    """ Input: path to image
    This functions reads the image and converts it into grayscale 
    return: gray image
    """
    img = cv2.imread(path)
    # this step is important for edge detection , reading the image in gray is not enough
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imggray = np.float32(imggray)
    return imggray

def calc_surrounding_keypoint(point_x_of_kp, point_y_of_kp, size):
    """Input: x and y of Keypoint and the Keypoint size 
    This function calculates the buttom right and top left corner for subwindow
    return: List with points for subwindow [[tl_x,br_x],[tl_y,br_y]]
    """
    # modulo calculation
    mod = lambda x,md: x%md      
    
    # floor calculation to stick with given size around the Keypoint
    dev = math.floor(size/2)
    
    pt_x_top_left = point_x_of_kp - dev
    pt_y_top_left = point_y_of_kp - dev
    
    pt_x_botom_right = point_x_of_kp + dev
    pt_y_botom_right = point_y_of_kp + dev
    
    # if the given size is odd create odd subwindow
    if mod(size,2) == 1:
        pt_x_botom_right = point_x_of_kp + dev + 1
        pt_y_botom_right = point_y_of_kp + dev + 1    
    
    return [np.uint8([pt_x_top_left,pt_x_botom_right]),np.uint8([pt_y_top_left,pt_y_botom_right])]

keypoints = [cv2.KeyPoint(15, 15, 11)]

# test for all test images
# preprocessing and load the images
# os.chdir("/Users/Kostja/Desktop/Master/Sem 5 (19:20 WiSe)/Learning from Images/Assignments/lfi-02")
img = glob.glob('./images/hog_test/*.jpg')
images = [load_image_and_cvt(img[i]) for i in range(len(img))]

descriptor = [compute_simple_hog(images[i], keypoints, img[i]) for i in range(len(images))]

