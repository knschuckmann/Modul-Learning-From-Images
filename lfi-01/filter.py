import numpy as np
import cv2


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


def convolution_2d(img, kernel):
    """
    Computes the convolution between kernel and image

    :param img: grayscale image
    :param kernel: convolution matrix - 3x3, or 5x5 matrix
    :return: result of the convolution
    """
    # TODO write convolution of arbritrary sized convolution here
    # Hint: you need the kernelsize

    offset = int(kernel.shape[0]/2)
    # make an offset and replicate the very last element for offste times
    img = cv2.copyMakeBorder(img, offset,offset,offset,offset, cv2.BORDER_REPLICATE )
    
    length_img = len(img)
    
    newimg = np.zeros(img.shape)
    
    for y in range(offset,length_img-offset-1):
        for x in range(offset,length_img-offset-1):
            grad_x = sum((kernel * img[y-offset:y+offset+1,x-offset:x+offset+1]).sum(axis = 0))
            # grad_y = sum((kernel.transpose() * img[y-offset:y+offset+1,x-offset:x+offset+1]).sum(axis = 0))
            #newimg[y,x] = np.sqrt(grad_x**2 + grad_y**2)
            newimg[y,x] = grad_x
            
    # YOUR CODE HERE

    return newimg


if __name__ == "__main__":
    # 1. load image in grayscale
    img = cv2.imread("Lenna.png",0)

    # 2. convert image to 0-1 image (see im2double)
    img_double = im2double(img)

    # image kernels
    # https://www.youtube.com/watch?v=W7OpxFbrD84
    sobelmask_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelmask_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gk = make_gaussian(11)


    # python works faster with lists
    # img_double1 = img_double.tolist()
    # 3 .use image kernels on normalized image
    #sobel_x = convolution_2d(img_double, sobelmask_x)
    #sobel_y = convolution_2d(img_double, sobelmask_y)
    
    
    
    # 4. compute magnitude of gradients
    #mog = np.sqrt( sobel_x**2 + sobel_y**2)
    
    
    # better approach and faster one only for sobel in x and y direction possible
    grad_x = convolution_2d(img_double, sobelmask_x)
    grad_y = convolution_2d(img_double, sobelmask_y)
    # gaus = convolution_2d(img_double, gk)
    # mog = gaus
    mog = np.sqrt(grad_x**2 + grad_y**2)
    
    
    # cs2.imshow sobel_x and sobel_y then comment out also the calsculation of sobel_ and sobel_y 
    # newimg[x,y] = grad_x comment out 
    
    # Show resulting images
    #cv2.imshow("sobel_x", sobel_x)
    #cv2.imshow("sobel_y", sobel_y)
    cv2.imshow("mog", mog)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
