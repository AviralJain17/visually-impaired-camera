import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
#for detecting the shape
def shape_detection(cont):
    
    a = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.04 * a, True)
    l=len(approx)
    if l==3 :# detect triangle
        return 'triangle'
    elif l==4 :#detect rectangle or square
        (x, y, w, h) = cv2.boundingRect(approx)
        ratio = w / float(h)
        if ratio>=0.9 and ratio<=1.1:
            return 'square'
        else :
            return'rectangle'
            
    elif l>=6 and l<=30:# detecting circle
        return 'circle'
#detecting colour
def colour_detector(frame,cont):
    
    blue= [255,0,0]
    green=[0,255,0]
    red=[0,0,255]
    

    mask = np.zeros(frame.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [cont], -1, 255, -1)
    mean = cv2.mean(frame,mask=mask)[:3]

    
    min_d_color= [np.inf,None]
    color=[blue,green,red]
    color_l=['blue','green','red']
    for i in range(3):
         d=dist.euclidean(color[i], mean)
         if d<min_d_color[0]:
            min_d_color[0]=d
            min_d_color[1]=color_l[i]
    return min_d_color[1]
cap=cv2.VideoCapture(0)
while(1):
    ret,frame=cap.read() 
    if ret== True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur=cv2.GaussianBlur(gray,(5,5),0)
        _,thresh = cv2.threshold(blur,150,255,cv2.THRESH_BINARY_INV) 
        kernel = np.ones((3, 3),np.uint8)
        dilated_image = cv2.dilate(thresh, kernel, iterations = 1)
        contours,h = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        

        for i in contours:
         if cv2.arcLength(i, True) <300: 
             continue
         if cv2.contourArea(i) > 100 : 
        
            shape=shape_detection(i)
            if shape in ['rectangle','square','circle','triangle']:
             
             color=colour_detector(frame,i) 
             epsilon = 0.001 * cv2.arcLength(i, True)
             approx = cv2.approxPolyDP(i, epsilon, True)
             M = cv2.moments(i)                               #cx,cy is center 
             cX = int(M["m10"] /( M["m00"]))
             cY = int(M["m01"] /( M["m00"]))
             cv2.drawContours(frame,[approx], -1, (0, 255, 0), 3) 
             cv2.putText(frame, f'shape:{shape} color:{color}',(cX , cY ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
             cv2.imshow('FRAME',frame)
             cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF==ord('q'):  #clicking q for closing camera
            break
    
cv2.destroyAllWindows()
cap.release()

