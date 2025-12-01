import cv2
import pickle
import numpy as np
from cvzone.HandTrackingModule import HandDetector

print("Loading model...")
try:
    with open('model.p', 'rb') as f:
        model_dict = pickle.load(f)
        if isinstance(model_dict, dict) and 'model' in model_dict:
            model = model_dict['model']
        else:
            model = model_dict
except FileNotFoundError:
    print("Error: model.p not found. Make sure it's in the same folder.")
    exit()

cap = cv2.VideoCapture(1) 
detector = HandDetector(maxHands=1)

print("Starting camera...")

while True:
    success, img = cap.read()
    if not success:
        break
        
    img = cv2.flip(img, 1) 
    imgOutput = img.copy()
    
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        lmList = hand['lmList']
        x, y, w, h = hand['bbox']

        
        normalized_landmarks = []
        for lm in lmList:
            
            norm_x = (lm[0] - x) / w
            norm_y = (lm[1] - y) / h
            normalized_landmarks.extend([norm_x, norm_y])

       
        try:
            prediction = model.predict([normalized_landmarks])
            class_name = prediction[0]
            
            probabilities = model.predict_proba([normalized_landmarks])
            confidence = np.max(probabilities)

            
            if confidence > 0.7:
                cv2.rectangle(imgOutput, (x, y - 50), (x + w, y), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, f'{class_name} {int(confidence*100)}%', (x + 5, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x, y), (x + w, y + h), (255, 0, 255), 2)
            
        except Exception as e:
            pass

    cv2.imshow("Sign Language Detector", imgOutput)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()