
import numpy as np
import cv2

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap=cv2.VideoCapture(0)
Id = input('enter your id')
sampleNum = 0
while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        # incrementing sample number
        sampleNum = sampleNum + 1
        # saving the captured face in the dataset folder
        cv2.imwrite("dataSet/User." + Id + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])

        cv2.imshow('frame', img)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif sampleNum > 20:
        break

cap.release()
cv2.waitKey()
cv2.destroyAllWindows()





