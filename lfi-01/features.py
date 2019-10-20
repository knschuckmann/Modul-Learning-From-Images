"""
SIFT (Scale Invariant Feature Transform)
1. Scale-space Extrema Detection
2. Keypoint Localization
3. Orientation Assignment
4. Keypoint Descriptor
5. Keypoint Matching

"""
import cv2
#create SIFT Object
sift = cv2.xfeatures2d.SIFT_create()
cap = cv2.VideoCapture(0)
cv2.namedWindow('Learning from images: SIFT feature visualization')
while True:
    # 1. read each frame from the camera (if necessary resize the image)
    #    and extract the SIFT features using OpenCV methods
    #    Note: use the gray image - so you need to convert the image
    # 2. draw the keypoints using cv2.drawKeypoints
    #    There are several flags for visualization - e.g. DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS

    # close the window and application by pressing a key
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    
    # detect Keypoints like described and compute descriptors
    kp, des = sift.detectAndCompute(frame, None)
    
    frame = cv2.drawKeypoints(frame,kp,frame, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)    
    # wait for key and switch to mode
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('q'):
        break
    
    cv2.imshow('Learning from images: SIFT feature visualization', frame)
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()