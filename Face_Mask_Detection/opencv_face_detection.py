import cv2
import numpy as np
from tensorflow.keras.models import load_model

cap = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model("model.h5")

while True:
    ret, frame = cap.read() #讀取每一幀

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceRects = cascade.detectMultiScale(grey, scaleFactor = 1.1, minNeighbors = 3)
    for faceRect in faceRects:
        x, y, w, h = faceRect        
        face = frame.copy()[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = cv2.resize(face, (64, 64))
        data = face[np.newaxis]
        cv2.imshow('Face', face)
        predicts = model.predict(data)
        # predicts_classes = np.argmax(predicts, 1)
        print(predicts)
    cv2.imshow('Haar', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()