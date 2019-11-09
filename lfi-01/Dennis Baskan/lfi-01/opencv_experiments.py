"""
Kostja Comments:
    - Again you used the RGB but cv2 gives you BGR if you read the documentation and the lecture
    - Where did you get the Otsu Thereshold and why is the adaptive gaussian not really adaptive but only gaussian blurr
    - the print in the begining is really nice
    - I am not quite sure if thta what the prof wants if you do canny you wont get away and gaussian can be applied many time best to see when canny
    is on. The otsu should look different in my opinion.

"""
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
mode = 0
filtermode = ''
print("Color Spaces:\n1: HSV\n2: LAB\n3: YUV\n0: RGB\n\nFilters:\ng: Gaussian Blur\no: Otsu Thresholding\nc: Canny Edge\nx: No Filters\n\nq: Quit Application")
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # wait for key and switch to mode
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('1'):      # HSV
        mode = 1
    elif ch == ord('2'):    # LAB
        mode = 2
    elif ch == ord('3'):    # YUV
        mode = 3
    elif ch == ord('0'):    # RGB
        mode = 0
    elif ch == ord('g'):    # apply Gaussian filter
        filtermode += 'g'
    elif ch == ord('o'):    # apply Otsu Thresholding (only possible without Canny edge)
        filtermode = filtermode.replace('o','')
        filtermode = filtermode.replace('c','')
        filtermode += 'o'
    elif ch == ord('c'):    # apply Canny Edge Extraction
        filtermode = filtermode.replace('c','')
        filtermode += 'c'
    elif ch == ord('x'):    # remove all filters
        filtermode = ''

    # change color space
    if mode == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    elif mode == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    elif mode == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
                
    #print(filtermode)
    # apply filters in defined order
    for f in  filtermode:
        if f == 'g':
            frame = cv2.GaussianBlur(frame, (5,5), 0)
        elif f == 'o':
            # Otsu's thresholding (https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html#otsus-binarization)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            ret2,th2 = cv2.threshold(frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        elif f == 'c':
            #Canny Edge  (https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html)
            frame = cv2.Canny(frame,100,200)
            
    # quit application
    if ch == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('frame', frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



'''
References:
    
    -OpenCV Docs:
        https://opencv-python-tutroals.readthedocs.io/en/latest/
    -Moodle files:
        https://lms.beuth-hochschule.de/mod/resource/view.php?id=469961
        
'''
