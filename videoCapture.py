import cv2
import numpy as np
import time
from test import prepareImage

def facialRecogniction(model, pca_components):
    # Load the pre-trained Haar cascade for face and eyes detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
    cam = cv2.VideoCapture(0)
    print("cam initializated")

    while cam.isOpened():
        ret, frame = cam.read() 

        if ret:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(70,70)) 

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) 

                #get eyes 
                gray_frame = gray[y:y+h,x:x+w]
                color_frame = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(gray_frame, scaleFactor = 1.02, minNeighbors = 7, minSize = (40,40), maxSize=(60,60))

                for e in eyes:
                    eye = is_eye(e, gray_frame)
                    if eye:
                        xe, ye, we, he = e
                        eye = gray_frame[ye:ye+he, xe:xe+we]

                        img = prepareImage(eye, pca_components)
                        prediction = model.predict(img)
                        if prediction == 1:
                            cv2.rectangle(color_frame, (xe, ye), (xe+we, ye+he), (0, 0, 255), 2)
                            print("Eye is closed")
                        else:
                            cv2.rectangle(color_frame, (xe, ye), (xe+we, ye+he), (0, 255, 0), 2)
                            print("Eye is opened")
                    else: 
                        continue
                    
        cv2.imshow("Video Capture", frame)
        time.sleep(0.05)
        
        if cv2.waitKey(1) == ord('q'):
            break
        
    cam.release() 
    cv2.destroyAllWindows()

def is_eye(region, image):
    x, y, w, h = region
    return y+h < 0.6 * image.shape[0] 