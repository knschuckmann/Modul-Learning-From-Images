"""
Kostja Comments:
    is the Outimage in drawKeypoints neccesary because u use already =  

"""
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cv2.namedWindow('Learning from images: SIFT feature visualization')
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # get dimensions of frame
    m,n,c = frame.shape
    # resize if necessary
    scale = 1.0
    frame = cv2.resize(frame,(int(n*scale),int(m*scale)))
    
    #read single image
    #frame = cv2.imread('south-korea--geunjeongjeon-gyeongbokgung.jpg')
    #frame = cv2.imread('heart.jpg')
    
    # wait for key to quit application
    ch = cv2.waitKey(1) & 0xFF

    # RGB to Gray
    gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Apply SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)

    # draw Keypoints
    frame = cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,outImage=frame)
    
    # save single image
    #cv2.imwrite('sift_keypoints.jpg',frame)

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
    -https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html
    -images: 
        https://cdn.cnn.com/cnnnext/dam/assets/150421112835-beautiful-south-korea--geunjeongjeon-gyeongbokgung-full-169.jpg
        https://cdn.britannica.com/88/22488-050-8AAD90B0/conduction-heart-individuals-sinoatrial-node-pacemaker-cells.jpg

'''


