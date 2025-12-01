import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import csv
import os

cap = cv2.VideoCapture(1)
detector = HandDetector(maxHands=1)

current_label = "Love you"

file_path = "SignLanguageData.csv"

if not os.path.exists(file_path):
    with open(file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        
        headers = ["label"]
        for i in range(21):
            headers.extend([f"x{i}", f"y{i}"])
        writer.writerow(headers)

print(f"Press 's' to save data for: {current_label}")

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    
    hands, img = detector.findHands(img, draw=True) 

    if hands:
        lmList = hands[0]['lmList'] 
        
        
        x, y, w, h = hands[0]['bbox']
        
        normalized_landmarks = []
        for lm in lmList:
            
            norm_x = (lm[0] - x) / w
            norm_y = (lm[1] - y) / h
            normalized_landmarks.extend([norm_x, norm_y])

        key = cv2.waitKey(1)
        if key == ord('s'):
            with open(file_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([current_label] + normalized_landmarks)
            print(f"Data Saved for {current_label}")

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()