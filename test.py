import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import csv
import os

cap = cv2.VideoCapture(1) 
detector = HandDetector(maxHands=2) 

current_label = "You" #isme photo ka label dalo
file_path = "Dhruv.csv"

def get_normalized_landmarks(hand):
    lmList = hand['lmList']
    x, y, w, h = hand['bbox']
    normalized = []
    for lm in lmList:
        norm_x = (lm[0] - x) / w
        norm_y = (lm[1] - y) / h
        normalized.extend([norm_x, norm_y])
    return normalized

if not os.path.exists(file_path):
    with open(file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        
        headers = ["label"]

        for i in range(21):
            headers.extend([f"h1_x{i}", f"h1_y{i}"])

        for i in range(21):
            headers.extend([f"h2_x{i}", f"h2_y{i}"])
            
        writer.writerow(headers)

print(f"Press 's' to save data for: {current_label}")

while True:
    success, img = cap.read()
    if not success:
        break
    
    hands, img = detector.findHands(img, draw=True) 

    if hands:
        data_row = []
        
        hand1 = hands[0]
        data_row.extend(get_normalized_landmarks(hand1))
        
        if len(hands) == 2:
            hand2 = hands[1]
            data_row.extend(get_normalized_landmarks(hand2))
        else:
            
            data_row.extend([0] * 42)

        key = cv2.waitKey(1)
        if key == ord('s'):
            with open(file_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([current_label] + data_row)
            print(f"Data Saved for {current_label} (Hands detected: {len(hands)})")

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()