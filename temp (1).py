import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
#detector_params.minArea = 60
detector = cv2.SimpleBlobDetector_create(detector_params)
C_points=((10,10),(10,375),(10,740),(375,10),(375,375),(375,740),(740,10),(740,375),(740,740))
Rx_points=[]
Ry_points=[]


def mapping(x,y,xmax,xmin,ymax,ymin):
    Xs=int((xmax-x)*750/(xmax-xmin))
    Ys=int((y-ymin)*750/(ymax-ymin))
    screen = np.zeros((750,750,3), np.uint8)
    cv2.waitKey(100)
    cv2.circle(screen,(Xs,Ys),3,(0,255,0))
    cv2.imshow('Screen',screen)
    cv2.moveWindow('calibration window',50,0)

def calibration(x,frame):
    img = np.zeros((750,750,3), np.uint8)
    cv2.circle(img,x,3,(0,255,0))
    cv2.imshow('calibration window',img)
    cv2.moveWindow('calibration window',50,0)
    cv2.waitKey(1500)
    eye=frame[100:140,260:310]
    threshold = cv2.getTrackbarPos('threshold', 'image')
    keypoints = blob_process(eye, threshold, detector)
    for keypoint in keypoints:
        x=keypoint.pt[0]
        y=keypoint.pt[1]
        Rx_points.append(x)
        Ry_points.append(y)

def blob_process(img, threshold, detector):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    e = cv2.getTrackbarPos('erode', 'image')
    d = cv2.getTrackbarPos('dilate', 'image')
    img = cv2.erode(img, None, iterations=1)
    img = cv2.dilate(img, None, iterations=3)
    img = cv2.medianBlur(img, 3)
    keypoints = detector.detect(img)
    cv2.imshow('blob',img)
    cv2.moveWindow('blob',700,200)
    return keypoints


def nothing(defult=None):
    pass



def main():
    cap = cv2.VideoCapture(1)
    cv2.namedWindow('image')
    cv2.namedWindow('blob')
    cv2.moveWindow('blob',100,200)
    cv2.resizeWindow('blob',420,420)
    cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
    cv2.createTrackbar('erode', 'image', 0, 10, nothing)
    cv2.createTrackbar('dilate', 'image', 0, 10, nothing)
    f=0
    while True:
        _, frame = cap.read()
        #process(frame)
        eye=frame[100:140,260:310]   
        threshold = cv2.getTrackbarPos('threshold', 'image')
        keypoints = blob_process(eye, threshold, detector)
        for keypoint in keypoints:
            #print(eye.shape[:])
            x=keypoint.pt[0]
            y=keypoint.pt[1]
            if cv2.waitKey(1) & 0xFF == ord('s'):
                print(x,y)
            if f==1:
                xmin=min(Rx_points)
                ymin=min(Ry_points)
                xmax=max(Rx_points)
                ymax=max(Ry_points)
                mapping(x,y,xmax,xmin,ymax,ymin)
            #cv2.circle(eye,(10,10),2,(255,0,0))
            eye = cv2.drawKeypoints(eye, keypoints, eye, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('image', frame)
        cv2.imshow('image2', eye)
        cv2.moveWindow('image2',700,400)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            for i in C_points:
                _, frame = cap.read()
                calibration(i,frame)
            f=1
            cv2.destroyWindow('calibration window')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

main()