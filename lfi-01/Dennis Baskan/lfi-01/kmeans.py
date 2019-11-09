"""
Kostja Commenst:
    - I am not quite sure if the distance shouldnt be implemented by ouself without any help of np etc.
    - what is zip(* ....) mean
    - shouldnt be closest cluster more like calculate the distance to all clusters and take the min out of that instead calc the dist to just one
    - why is there 2 for loops in kmeans one that runs only once without it it could work too. Is that te optimization for your code?
"""

import numpy as np
import cv2
import sys
from numpy.linalg import norm 
from numpy.random import choice as rand
############################################################
#
#                       KMEANS
#
############################################################

# implement distance metric - e.g. squared distances between pixels
def distance(a, b):
    return (norm(a-b) ** 2) #norm() returns Euclidean Distance by default
    

# k-means works in 3 steps
# 1. initialize
# 2. assign each data element to current mean (cluster center)
# 3. update mean
# then iterate between 2 and 3 until convergence, i.e. until ~smaller than 5% change rate in the error

def update_mean(img, clustermask,result):
    """This function should compute the new cluster center, i.e. numcluster mean colors"""
    # YOUR CODE HERE
    for cluster in range(numclusters):
        cpix = (clustermask == cluster)[:,:,0]
        if np.count_nonzero(cpix) > 0:
            current_cluster_centers[cluster] = np.mean(img[cpix],axis=0)
            result[cpix] = cluster_colors[cluster]
                    
    return current_cluster_centers,result


def closest_cluster(pixel,current_cluster_centers):
    """
    Return: [Distance to closest cluster,Cluster ID]
    """
    dists = [distance(pixel,cluster[0]) for cluster in current_cluster_centers]
    cluster = np.argmin(dists)
    return [dists[cluster],cluster]

    
def assign_to_current_mean(img, result, clustermask):
    """The function expects the img, the resulting image and a clustermask.
    After each call the pixels in result should contain a cluster_color corresponding to the cluster
    it is assigned to. clustermask contains the cluster id (int [0...num_clusters]
    Return: the overall error (distance) for all pixels to there closest cluster center 
            (mindistance px - cluster center).
    """
    # closest distance to cluster
    distances,clusters = zip(*[closest_cluster(img[yp,xp],current_cluster_centers) for yp in range(h1) for xp in range(w1)])

    # update cluster mask
    clustermask = np.asarray(clusters).reshape([h1,w1,1])
    
    # calculate overall distance
    overall_dist = np.sum(distances)
    
    return overall_dist,clustermask


def initialize(img):
    """inittialize the current_cluster_centers array for each cluster with a random pixel position"""
    # initialize random clusters    
    current_cluster_centers[:,0,0] = rand(range(256),numclusters)
    current_cluster_centers[:,0,1] = rand(range(256),numclusters)
    current_cluster_centers[:,0,2] = rand(range(256),numclusters)
    

def kmeans(img,runs=1):
    """Main k-means function iterating over max_iterations and stopping if
    the error rate of change is less then 2% for consecutive iterations, i.e. the
    algorithm converges. In our case the overall error might go up and down a little
    since there is no guarantee we find a global minimum.
    """
    best_dist = sys.float_info.max
    for run in range(runs):
        print("Run: "+str(run))
        max_iter = 10
        max_change_rate = 0.02
        dist = sys.float_info.max
    
        clustermask = np.zeros((h1, w1, 1), np.uint8)
        result = np.zeros((h1, w1, 3), np.uint8)
    
        # initializes each pixel to a cluster
        initialize(img)
        
        # iterate for a given number of iterations or if rate of change is very small
        for it in range(max_iter):    
            #assign randomly clusters
            err,clustermask = assign_to_current_mean(img,result,clustermask)
            
            #calculate relative error
            change_rate = np.round(((dist-err)/dist),3)
            print("Iteration: %i\tError(absolute): %.3f\tError(relative): %.3f%%" %(it,err,change_rate*100))
                
            # update colors according to clustermask
            current_cluster_centers, result = update_mean(img,clustermask,result)
            dist = err
            
            if change_rate < max_change_rate:
                break
            
        if best_dist > dist:
            best_dist = dist
            best_result = result
            
    print("\nError(absolute) for best run: %.3f\n" %(best_dist))

    return best_result


# num of cluster
numclusters = 3
# corresponding colors for each cluster
cluster_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 255, 255], [0, 0, 0], [128, 128, 128]]
# initialize current cluster centers (i.e. the pixels that represent a cluster center)
current_cluster_centers = np.zeros((numclusters, 1, 3), np.float32)

# load image
imgraw = cv2.imread('Lenna.png')
scaling_factor = 0.5
imgraw = cv2.resize(imgraw, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

# compare different color spaces and their result for clustering
for mode in range(3):
    image = imgraw
    h1, w1 = image.shape[:2]
    
    # change color space
    if mode == 0:
        print("\nHSV")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif mode == 1:
        print("\nLAB")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    else:
        print("\nRGB")
                    
    # execute k-means over the image
    # it returns a result image where each pixel is color with one of the cluster_colors
    # depending on its cluster assignment
    res = kmeans(image,2)
    
    h1, w1 = res.shape[:2]
    h2, w2 = image.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = res
    vis[:h2, w1:w1 + w2] = image
    
    cv2.imshow("Color-based Segmentation Kmeans-Clustering", vis)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
References:
    -https://lms.beuth-hochschule.de/pluginfile.php/893711/mod_resource/content/0/LFI-02-ImageProcessing.pdf
    -https://docs.scipy.org/doc/numpy/reference/
    -https://opencv-python-tutroals.readthedocs.io/en/latest/
"""