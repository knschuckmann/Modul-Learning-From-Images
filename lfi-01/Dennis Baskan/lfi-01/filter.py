"""
Kostja Comments: 
    - Can u explain the ravel() function and why you dont give it a order?? in im2double
    - when using convolution2d is it fastser to create the variables and not claculate with calculated values?
    - Very nice codestyle
    - Is Gausian Vlur required before aplying the sobel?
    - why do you use angle
"""
import numpy as np
import cv2
from time import time as t
import timeit

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


def make_gaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    k = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    return k / np.sum(k)

def convolution_2d(img, kernel,mode='padding'):
    """
    Computes the convolution between kernel and image

    :param img: grayscale image
    :param kernel: convolution matrix - 3x3, or 5x5 matrix
    :return: result of the convolution
    """
    k = kernel.shape[0]
    offset = int(k/2)
    imgm,imgn = img.shape
        
    if mode == 'padding':
        #img = np.pad(img,[(moff,moff2),(noff,noff2)])
        img = np.pad(img,offset)
    
    #calculate convolution    
    m = [np.sum(img[i:i+k,j:j+k] * kernel) for i in range(imgm) for j in range(imgn)]
    newimg = np.matrix(m).reshape((imgm,imgn))

    return newimg

if __name__ == "__main__":

    # 1. load image in grayscale+scale
    frame = cv2.imread('heart.jpg',0)
    #frame = cv2.imread('Lenna.png',0)
    m,n = frame.shape[:2]
    s = 0.35
    frame = cv2.resize(frame,(int(n*s),int(m*s)))    
    
    # 2. convert image to 0-1 image (see im2double)
    frame = im2double(frame)

    # image kernels
    sobelmask_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelmask_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gk = make_gaussian(11)

    # 3. use image kernels on normalized image
    frame = convolution_2d(frame,gk)
    print("Gaussian Blur Done!")
    sobel_x = convolution_2d(frame,sobelmask_x)
    print("Sobel X Done!")
    sobel_y = convolution_2d(frame,sobelmask_y)
    print("Sobel Y Done!")

    # 4. compute magnitude of gradients
    mog = np.sqrt(np.multiply(sobel_x,sobel_x)+np.multiply(sobel_y,sobel_y))
    print("MOG Done!")
    angle = np.arctan2(sobel_y,sobel_x)
    print("Angle Done!")

    # Show resulting images
    cv2.imshow("sobel_x", sobel_x)
    cv2.imshow("sobel_y", sobel_y)
    cv2.imshow("mog", mog)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
'''
References:
    -https://lms.beuth-hochschule.de/pluginfile.php/781445/mod_resource/content/0/LFI-02-ImageProcessing.pdf
    -https://docs.scipy.org/doc/numpy/reference/
    -https://docs.python.org/2/library/timeit.html
    -https://joblib.readthedocs.io/en/latest/parallel.html
    
'''
np.matrix(4).reshape((2,2))
