import numpy as np
import cv2
import webbrowser
import os

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read('trainner\\trainner.yml')
id=0
#cv2.FONT_HERSHEY_SIMPLEX
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 2
thickness = 2
fontcolor = (5000 ,0 ,0)
c=0
d=0
e=0
while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        Id, conf = rec.predict(gray[y:y + h, x:x + w])
        if (Id == 1):
            Id = "Hamza"
            c+=1
            if c==1:
                webbrowser.open('https://www.instagram.com/hamza.nufc10/')
                break
            elif c==2:
                break
                c=0
        elif(Id==2):
            Id="Messi"
            e += 1
            if e == 1:
                webbrowser.open('https://www.instagram.com/leomessi/')
                break
            elif e == 2:
                break
                e = 0
        elif(Id==3):
            Id="Ronaldo"
            d += 1
            if d == 1:
                webbrowser.open('https://www.instagram.com/cristiano/')
                break
            elif d == 2:
                break
                d=0
        else:
            Id="unknown"
        cv2.putText(img, str(Id), (x,y+h), fontface,fontscale, fontcolor,thickness)

    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release(0)
cv2.waitKey(0)
cv2.destroyAllWindows()

