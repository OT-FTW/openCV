import cv2
import numpy as np
import face_recognition

imgRonaldo = face_recognition.load_image_file("D:\Dropbox\python\Student_attendence\imagesMain/Ronaldo.jpg")
imgRonaldo = cv2.cvtColor(imgRonaldo,cv2.COLOR_BGR2RGB)

#imgMessi = face_recognition.load_image_file("D:\Dropbox\python\Student_attendence\imagesMain/Messi.jpg")
#imgTrunp = face_recognition.load_image_file("D:\Dropbox\python\Student_attendence\imagesMain/Trump.jpg")
#imgBiden = face_recognition.load_image_file("D:\Dropbox\python\Student_attendence\imagesMain/Biden.jpg")

imgTest = face_recognition.load_image_file("D:\Dropbox\python\Student_attendence\ImageTest/t1.jpg")
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgRonaldo)[0]
encodeRonaldo = face_recognition.face_encodings(imgRonaldo)[0]
cv2.rectangle(imgRonaldo,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(200,20,10),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(200,20,10),2)


results = face_recognition.compare_faces([encodeRonaldo],encodeTest)
faceDis = face_recognition.face_distance([encodeRonaldo],encodeTest)
print(results,faceDis) #True or False

cv2.putText(imgTest,f'{results}{"  "}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)


cv2.imshow("Ronaldo",imgRonaldo)
cv2.imshow("test",imgTest)
cv2.waitKey(0)
