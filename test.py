import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(1)  
detector = HandDetector(maxHands=1)
offset = 20
imgsize = 300
counter = 0

folder = "/Users/prajwalbhosale/Documents/project/data/thankyou"
os.makedirs(folder, exist_ok=True) 

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  

    if not success:
        print("Camera not working")
        break

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imagewhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255

        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            cv2.imshow("image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        h_c, w_c = imgCrop.shape[:2]
        aspectratio = h_c / w_c

       
        if aspectratio > 1:
            k = imgsize / h_c
            wcal = math.ceil(k * w_c)
            imgResize = cv2.resize(imgCrop, (wcal, imgsize))
            wgap = math.ceil((imgsize - wcal) / 2)
            imagewhite[:, wgap:wgap + wcal] = imgResize

        else:
            k = imgsize / w_c
            hcal = math.ceil(k * h_c)
            imgResize = cv2.resize(imgCrop, (imgsize, hcal))
            hgap = math.ceil((imgsize - hcal) / 2)
            imagewhite[hgap:hgap + hcal, :] = imgResize

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imagewhite)

    cv2.imshow("image", img)

    key = cv2.waitKey(1)

    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imagewhite)
        print("Saved:", counter)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
