import pickle
import cv2
import numpy as np
import signal
import sys

pt00 = [13,71]
ptwh = [106,48]
ptw0 = [155,71]
pt0h = [57,48]
w,h = 200,200
pts1 = np.float32([pt00,ptw0,ptwh,pt0h])
pts2= np.float32([[0,0],[w,0],[w,h],[0,h]])

def signal_handler(sig, frame):
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Load images from pickle file
with open('coeffs_train.p', 'rb') as f:
    images = pickle.load(f)

# Display images one at a time
for i in range(len(images)):
    img = images[i]
    cv2.imshow('original', img)

    #apply transformation
    mat = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,mat,(w,h))

    #rotate the image 180 degrees
    dst = cv2.rotate(dst, cv2.ROTATE_180)
    #filp the image
    dst = cv2.flip(dst, 1)

    
    #apply gaussian blur
    dst = cv2.GaussianBlur(dst,(5,5),0)

    #sharpen the image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    dst = cv2.filter2D(dst, -1, kernel)

    #upscale the image
    dst = cv2.resize(dst, (0,0), fx=2, fy=2)




    # Display the transformed image
    cv2.imshow('transformed', dst)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        sys.exit(0)

# Close all windows
cv2.destroyAllWindows()

