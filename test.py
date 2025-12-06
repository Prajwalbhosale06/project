import cv2
import numpy as np
import os
import mediapipe as mp

# --- CONFIGURATION ---
DATA_PATH = os.path.join('MP_Data') 
# ADDED 'Idle' class for silence/nothing
actions = np.array(['Hello', 'NO']) 
no_sequences = 30 
sequence_length = 30

for action in actions: 
    os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    """
    Extracts coordinates relative to the wrist to make the model
    independent of your position in the room.
    """
    # 1. Pose (Keep raw for now as it anchors the body)
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33*4)
        
    # 2. Left Hand (Relative to Wrist)
    if results.left_hand_landmarks:
        wrist = results.left_hand_landmarks.landmark[0]
        lh = np.array([[res.x - wrist.x, res.y - wrist.y, res.z - wrist.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21*3)

    # 3. Right Hand (Relative to Wrist)
    if results.right_hand_landmarks:
        wrist = results.right_hand_landmarks.landmark[0]
        rh = np.array([[res.x - wrist.x, res.y - wrist.y, res.z - wrist.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21*3)
        
    return np.concatenate([pose, lh, rh])

cap = cv2.VideoCapture(1)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            window = [] 
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret: break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting {} #{}'.format(action, sequence), (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000) 
                else: 
                    cv2.putText(image, 'Collecting {} #{}'.format(action, sequence), (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                
                keypoints = extract_keypoints(results)
                window.append(keypoints)
                if cv2.waitKey(10) & 0xFF == ord('q'): break
            
            # SAVE ONLY IF VALID (30 frames)
            if len(window) == 30:
                npy_path = os.path.join(DATA_PATH, action, str(sequence))
                np.save(npy_path, np.array(window))

    cap.release()
    cv2.destroyAllWindows()