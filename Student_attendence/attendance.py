import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'D:\Dropbox\python\Student_attendence\imagesMain'
Images = []
ClassNames = []

myList = os.listdir(path)
print (myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    Images.append(curImg)
    ClassNames.append(os.path.splitext(cl)[0])

print(ClassNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('D:\Dropbox\python\Student_attendence\Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            daString = now.strftime('%H:%M')
            f.writelines(f'\n{name},{daString}')

encodeListKnown = findEncodings(Images)
print('Encoding complete')

cap = cv2.VideoCapture(2)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    for encodeFace , faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matcheIndex = np.argmin(faceDis)

        if matches[matcheIndex]:
            name = ClassNames[matcheIndex].upper()
            #print(name)
            y1,x2,y2,x1= faceLoc
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,255),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,255),cv2.FILLED)
            cv2.putText(img,name,(x1,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
            markAttendance(name)


    cv2.imshow('webcam',img)
    cv2.waitKey(1)

    
