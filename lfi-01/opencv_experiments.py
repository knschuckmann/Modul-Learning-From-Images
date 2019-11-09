"""
Konstantin Schuckmann
Experiments on OpenCV
colorspace changing etc
20.10.19
"""

"""
There are more than 150 color spaces in OpenCV
call function: cv2.cvtColor(input_image, flag)

what are the flags 
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print (flags)

How to set Threashhold
cv2.inRange(hsv, lower_blue, upper_blue)

YUV is for analog colored TV
LAB is the distance between colors 
RGB Red Green Blue color density
HSV similar to human recognition (Hue/Farbwert, saturation/ Farbsättingung, value/ Hellwert)

Ozu Threshhold
But consider a bimodal image (In simple words, bimodal image is an image 
whose histogram has two peaks). For that image, we can approximately take 
a value in the middle of those peaks as threshold value, right ? That is what
Otsu binarization does. So in simple words, it automatically calculates a 
threshold value from image histogram for a bimodal image. (For images which
are not bimodal, binarization won’t be accurate.)

Thershold 
Gaussian
cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst])

Canny Edges
1. It is a multi-stage algorithm and we will go through each stages.
2. Noise Reduction
   Since edge detection is susceptible to noise in the image, first step is to remove the noise in the image with a 5x5 Gaussian filter. We have already seen this in previous chapters.
3. Finding Intensity Gradient of the Image
4. Non-maximum Suppression
5. Hysteresis Thresholding
OpenCV puts all the above in single function, cv2.Canny(img, minVal, maxVal)

"""
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

neutral_given, addapt_gaussian_threashhold,addapt_ozu_threashhold,canny,hsv_flag,lab_flag,yuv_flag,gray_flag = [False]*8

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # wait for key and switch to mode
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('n'):
        neutral_given = not neutral_given
    #Convert BGR to HSV
    if ch == ord('h'):
        hsv_flag = not hsv_flag
    #Convert BGR to LABq
    if ch == ord('l'):
        lab_flag = not lab_flag
        #Convert BGR to YUV
    if ch == ord('y'):
        yuv_flag = not yuv_flag
    if ch == ord('b'):
        gray_flag = not gray_flag
    #b for Threshold the Scale needs to be grayscale
    if ch == ord('t'):
        addapt_ozu_threashhold = not addapt_ozu_threashhold
    if ch == ord('g'):
        addapt_gaussian_threashhold = not addapt_gaussian_threashhold3
    if ch == ord('c'):
        canny = not canny
    if ch == ord('q'):
        break

    if neutral_given:
        # just example code but still 
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
    if hsv_flag :
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    if lab_flag:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)
    if yuv_flag:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2YUV)
    if gray_flag:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if addapt_ozu_threashhold:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)   
        # otsu finds the perfect value itself, therefor it is not needed to define adaptive threshold for otsu
        ret, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_OTSU)
    if addapt_gaussian_threashhold:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst])
        frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    if canny:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.Canny(frame,100,200)
    # Display the resulting frame
    cv2.imshow('frame', frame)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
