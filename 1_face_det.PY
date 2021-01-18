import cv2
#qimport face_recognition
import numpy as np
import time
#import pickle
print(cv2.__version__)

timeMark=time.time()
fpsFilter=0
font=cv2.FONT_HERSHEY_SIMPLEX
cap=cv2.VideoCapture(0)

def make_res(width, height):
    cap.set(3,width)
    cap.set(4,height)

make_res(1000,750)

# load the cascade Classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret,frame=cap.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    #print(faces)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    # FPS calculation
    dt=time.time()-timeMark
    fps=1/dt
    fpsFilter=.95*fpsFilter+.05*fps
    #print(fps,'  ',fpsFilter)
    #print(ret)
    cv2.putText(frame,str(round(fps,1))+' fps',(0,30),font,1,(0,255,255),2)
    timeMark=time.time()

    # Frame Display
    cv2.imshow('cam',frame)
    cv2.moveWindow('cam',960,0)
    #cv2.imshow('gray',gray)
    #cv2.moveWindow('gray',960,0)
  
  
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()  
cv2.destroyAllWindows()
