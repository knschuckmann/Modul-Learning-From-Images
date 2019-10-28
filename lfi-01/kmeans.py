import numpy as np
import cv2
import math
import sys


############################################################
#
#                       KMEANS
#
############################################################

# implement distance metric - e.g. squared distances between pixels
# a and b should be pixels on their own 
def distance(a, b):
    pass
    # YOUR CODE HERE
    temp = 0
    # euclidean distance between pixels in 3 dim color spaces
    if a.shape == b.shape and a.shape == (3,) and b.shape == (3,):
        for num in range(0,3):
            temp = temp + (a[num] - b[num])**2
        temp = np.sqrt(temp)
        return temp
    else:
        print('the Picture does not have a 3 dimensional colorspace, or a picture is given instead of a single pixel.')
     
# k-means works in 3 steps
# 1. initialize
# 2. assign each data element to current mean (cluster center)
# 3. update mean
# then iterate between 2 and 3 until convergence, i.e. until ~smaller than 5% change rate in the error

def update_mean(img, clustermask):
    """This function should compute the new cluster center, i.e. numcluster mean colors"""
    
    numclusters = len(np.unique(clustermask))
    overall_dist = np.zeros(numclusters,np.float32)
    clustermas = img[:,:,3]
    
    for cluster in range(0,numclusters):
        for rgb in range(0,current_cluster_centers.shape[2]):
            current_cluster_centers[cluster,0,rgb] = np.mean(img[img[img[:,:,3] == cluster,4].astype(int),img[img[:,:,3] == cluster,5].astype(int)][:,rgb])    
                  
    return current_cluster_centers


def assign_to_current_mean(img, result, clustermask):
    """The function expects the img, the resulting image and a clustermask.
    After each call the pixels in result should contain a cluster_color corresponding to the cluster
    it is assigned to. clustermask contains the cluster id (int [0...num_clusters]
    Return: the overall error (distance) for all pixels to there closest cluster center (mindistance px - cluster center).
    """
    numclusters = len(np.unique(clustermask))
    overall_dist = np.zeros((clustermask.shape[0],clustermask.shape[0],numclusters),np.float32)
    newMat = np.zeros((img.shape[0],img.shape[1],img.shape[2]+3), np.float32)
    temp = np.zeros(3, np.float32)
    res = np.zeros(2, np.float32)
    for x in range(0, clustermask.shape[0]):
        for y in range(0, clustermask.shape[1]):
            newMat[x,y,:img.shape[2]] = img[x,y]
            for cluster in range(0,numclusters):
                if clustermask[x][y] == cluster:
                    result[x,y] = current_cluster_centers[cluster]         
                overall_dist[x,y,cluster] = distance(img[x,y],current_cluster_centers[cluster,0])
                temp[cluster] = temp[cluster] + overall_dist[x,y,cluster]
            clustermask[x,y] = overall_dist[x,y].tolist().index(overall_dist[x,y].min())
            newMat[x,y,img.shape[2]] = clustermask[x,y]
            newMat[x,y,img.shape[2]+1] = x
            newMat[x,y,img.shape[2]+2] = y
            
    res = [overall_dist,newMat, temp, result]
    # YOUR CODE HERE
    return res

def clustermask_assign(numclusters,clustermask):
    """The function exspects the number of clusters and a possible clustermask for the shape of the final clustermask.
    Prefered is one clustermask assigned with only zeros.
    Return: a randomly assigned clustermask matrix for further calculations
    """
    clustermask = np.random.randint(0,numclusters,(clustermask.shape[0],clustermask.shape[1],clustermask.shape[2]))

    return clustermask
    

def initialize(img, numclusters):
    """inittialize the current_cluster_centers array for each cluster with a random pixel position"""
    # set seed for reproducability
    np.random.seed(200)
    
    # calc random cluster centers from standard normal dist args:1) number of clust variables 2) numb of clusters 3) numb of pixel to define a cluster
    #current_cluster_centers = np.concatenate( [np.random.randint(0,img.shape[0], (1,numclusters,2))[0],np.random.randint(0,img.shape[2], (1,numclusters,1))[0] ], axis = 1)
    if 'int' in img.dtype.name:
        # maybe accessing only two dimensions so the pixel contains all 3 values for rgb only for 3 dim : so rgb or hsv or lab
        # could be a color 
        current_cluster_centers = np.random.randint(0,img.max(), (numclusters,1,3))
        # or also a point to a pixel
        #current_cluster_centers = np.random.randint(0,img.shape[0], (1,numclusters,2))
        #img[current_cluster_centers.tolist()[0][0][0],current_cluster_centers.tolist()[0][0][1]]
    elif 'double' or 'float' in img.dtype.name:
        current_cluster_centers = np.random.randn(numclusters,1,3)

    return current_cluster_centers

def kmeans(img):
    """Main k-means function iterating over max_iterations and stopping if
    the error rate of change is less then 2% for consecutive iterations, i.e. the
    algorithm converges. In our case the overall error might go up and down a little
    since there is no guarantee we find a global minimum.
    """
    max_iter = 10
    max_change_rate = 0.02
    dist = sys.float_info.max
    
    clustermask = np.zeros((h1, w1, 1), np.uint8)
    result = np.zeros((h1, w1, 3), np.uint8)
    
    for iter in range(0,4):
        
        dist2, res, chan, result = assign_to_current_mean(img, result, clustermask)
        
        current_cluster_centers = update_mean(res, clustermask)
        
        if dist/chan.any() <= max_change_rate:
            break


    # initializes each pixel to a cluster
    # iterate for a given number of iterations or if rate of change is
    # very small
    # YOUR CODE HERE

    return result

def im2double(im):
    """
    Converts uint image (0-255) to double image (0.0-1.0) and generalizes
    this concept to any range.

    :param im:
    :return: normalized image
    """
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out


# num of cluster
numclusters = 3
# corresponding colors for each cluster
cluster_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 255, 255], [0, 0, 0], [128, 128, 128]]
# initialize current cluster centers (i.e. the pixels that represent a cluster center)
current_cluster_centers = np.zeros((numclusters, 1, 3), np.float32)

# load image
imgraw = cv2.imread('./images/Lenna.png')
scaling_factor = 0.5
imgraw = cv2.resize(imgraw, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

# compare different color spaces and their result for clustering
# YOUR CODE HERE or keep going with loaded RGB colorspace img = imgraw
image = imgraw
h1, w1 = imgraw.shape[:2]

cv2.imshow('new',imgraw)
# execute k-means over the image
# it returns a result image where each pixel is color with one of the cluster_colors
# depending on its cluster assignment
res = kmeans(image)

h1, w1 = res.shape[:2]
h2, w2 = image.shape[:2]
vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
vis[:h1, :w1] = res
vis[:h2, w1:w1 + w2] = image

cv2.imshow("Color-based Segmentation Kmeans-Clustering", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()


initialize(imgraw,numclusters)


distance(img[2,1], result[2,1])
