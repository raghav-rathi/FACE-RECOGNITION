import cv2
import numpy as np

p=r'F:\Face Recog\haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(p)
cam=cv2.VideoCapture(0);

id=int(input('enter user id'))
sampleNum=0;
while(True):
    ret, img = cam.read();
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(img, 1.3, 5)
    for (x,y,w,h) in faces:
        sampleNum=sampleNum+1;
        pth=r"F:\Face Recog\dataset\User"
        cv2.imwrite(pth+'//'+str(id)+"."+str(sampleNum)+".jpg",img[y:y+h,x:x+h])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow("Face",img);
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if(sampleNum>20):
                break
cam.release()
cv2.destroyALLWindows()
