import numpy as np
import cv2
import webbrowser

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read('trainner\\trainner.yml')
id=0
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1.5
fontcolor = (5000 ,0 ,0)
while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        Id, conf = rec.predict(gray[y:y + h, x:x + w])
        if (Id == 1):
            Id = "Hamza"
        elif(Id==2):
            Id="Messi"
        elif(Id==3):
            Id="Ronaldo"
        else:
            Id="unknown"
        cv2.putText(img, str(Id), (x,y+h), fontface, fontscale, fontcolor)

    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release(0)
cv2.waitKey(0)
cv2.destroyAllWindows()

