import cv2
import numpy as np

C_points=((10,10),(10,375),(10,740),(375,10),(375,375),(375,740),(740,10),(740,375),(740,740))
R_points=[]




def calibration(x):
   img = np.zeros((750,750,3), np.uint8)
   cv2.circle(img,x,3,(0,255,0))
   cv2.imshow('calibration window',img)
   cv2.moveWindow('calibration window',500,0)
   cv2.waitKey(1500)
   
for i in C_points:
    calibration(i)
    







			
     

   




