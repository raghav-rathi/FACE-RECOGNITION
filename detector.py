
import cv2
import numpy as np
p=r'F:\Face Recog\haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(p)
cam=cv2.VideoCapture(0);
recognizer=cv2.face.LBPHFaceRecognizer_create();
recognizer.read(r'F:\Face Recog\recognizer/trainningData.yml')
id=0
#font=cv2.FONT_HERSHEY_SIMPLEX,5,1,0,4
font=cv2.FONT_HERSHEY_SIMPLEX
while(True):
    ret, img = cam.read();
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(img, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf=recognizer.predict(gray[y:y+h,x:x+w])
        print(id)
        if(id==1):
            id="Rakshak"
     
        else:
            id="Unknown"
        cv2.putText(img,str(id),(x,y+h),font,2,(255,255,255));
        cv2.imshow('frame',img)
    if(cv2.waitKey(1) == ord('q')):
        break
cam.release()
cv2.destroyALLWindowns()