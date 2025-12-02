import cv2
import pickle
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# 1. Load the trained model
print("Loading model...")
try:
    with open('model.p', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: model.p not found. Make sure you ran train.py first.")
    exit()

# 2. Initialize Camera and Detector
cap = cv2.VideoCapture(1) # Change to 0 if using internal webcam
detector = HandDetector(maxHands=2) # Must detect 2 hands

# Helper function to normalize landmarks (Matches data collection logic)
def get_normalized_landmarks(hand):
    lmList = hand['lmList']
    x, y, w, h = hand['bbox']
    normalized = []
    for lm in lmList:
        # Normalize coordinates relative to the bounding box
        norm_x = (lm[0] - x) / w
        norm_y = (lm[1] - y) / h
        normalized.extend([norm_x, norm_y])
    return normalized

print("Starting camera...")

while True:
    success, img = cap.read()
    if not success:
        break
        
    img = cv2.flip(img, 1) 
    imgOutput = img.copy()
    
    # Detect Hands
    hands, img = detector.findHands(img, draw=True)

    if hands:
        # Prepare the data row (Must be 84 features long to match training)
        data_aux = []
        x_avg = 0
        y_avg = 0
        
        # --- PROCESS HAND 1 ---
        hand1 = hands[0]
        data_aux.extend(get_normalized_landmarks(hand1))
        
        # Calculate position for label (above hand 1)
        x1, y1, w1, h1 = hand1['bbox']
        x_avg = x1
        y_avg = y1

        # --- PROCESS HAND 2 (or Pad with Zeros) ---
        if len(hands) == 2:
            hand2 = hands[1]
            data_aux.extend(get_normalized_landmarks(hand2))
        else:
            # Critical: If model was trained on 2 hands, we MUST provide 84 inputs.
            # If only 1 hand is seen, we pad the remaining 42 inputs with zeros.
            data_aux.extend([0] * 42)

        try:
            # Predict
            prediction = model.predict([data_aux])
            class_name = prediction[0]
            
            # Get Confidence (Probability)
            probabilities = model.predict_proba([data_aux])
            confidence = np.max(probabilities)

            # Display Result if confidence is high enough
            if confidence > 0.7:
                # Draw a box with the label
                cv2.rectangle(imgOutput, (x_avg, y_avg - 50), (x_avg + 150, y_avg), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, f'{class_name} {int(confidence*100)}%', (x_avg + 5, y_avg - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
        except Exception as e:
            # This catches errors if the data shape doesn't match the model
            print(f"Prediction Error: {e}")
            pass

    cv2.imshow("Sign Language Detector", imgOutput)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()