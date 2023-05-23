import pickle
import cv2
import numpy as np
import signal
import sys

def signal_handler(sig, frame):
    cv2.destroyAllWindows()
    sys.exit(0)

def transformPrespective(img):
    #works well with vid_3.mp4
    pt00 = [0,540-100]
    ptwh = [230,260]
    ptw0 = [757,260]
    pt0h = [928,540-100] 
    w,h = 400,400
    pts1 = np.float32([pt00,ptwh,ptw0,pt0h])
    pts2= np.float32([[0,0],[w,0],[w,h],[0,h]])
    #apply transformation
    mat = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,mat,(w,h))
    #rotate the image 90 degrees
    dst = cv2.rotate(dst, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #return transformed image
    return dst

def edgeDetect(img):
    #apply edge detection on the image
    edges = cv2.Canny(img,100,200)
    #return edge detected image
    return edges

def FindStraightLines(img):
    #create a 400 by 400 rgb image 
    LineImage = np.zeros((400,400,3), np.uint8)
    #find straight lines in the image
    lines = cv2.HoughLinesP(img, 1, np.pi/180, 100, minLineLength=50, maxLineGap=1)
    #draw lines on the image
    try:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(LineImage,(x1,y1),(x2,y2),(255,0,0),5)
    except:
        pass
    #return image with lines
    return LineImage#, lines

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    #load video
    cap = cv2.VideoCapture('vids/vid_3.mp4')
                    
    #show image step by step
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            #resize frame to half size
            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            #apply transformation
            frame = FindStraightLines(edgeDetect(transformPrespective(frame)))

            #show frame
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif cv2.waitKey(1) & 0xFF == ord('s'):
                cv2.imwrite('frame.jpg',frame)