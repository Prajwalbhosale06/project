import cv2
import numpy as np
import os
import mediapipe as mp
import time

# --- CONFIGURATION ---
DATA_PATH = os.path.join('MP_Data') 

# All possible actions
ALL_ACTIONS = ['Hello', 'NO']

# ===== SELECT WHICH ACTION TO COLLECT =====
print("\n" + "="*60)
print("SELECTIVE DATA COLLECTION")
print("="*60)
print("\nAvailable actions:")
for idx, action in enumerate(ALL_ACTIONS):
    # Check how many sequences already exist
    action_path = os.path.join(DATA_PATH, action)
    existing_count = 0
    if os.path.exists(action_path):
        existing_count = len([f for f in os.listdir(action_path) if f.endswith('.npy')])
    print(f"  {idx + 1}. {action} (currently has {existing_count} sequences)")

print("\nWhich action do you want to collect?")
choice = input("Enter number (1-{}): ".format(len(ALL_ACTIONS)))

try:
    action_idx = int(choice) - 1
    if action_idx < 0 or action_idx >= len(ALL_ACTIONS):
        print("❌ Invalid choice!")
        exit()
except:
    print("❌ Invalid input!")
    exit()

selected_action = ALL_ACTIONS[action_idx]

# Ask how many sequences to collect
print(f"\nYou selected: '{selected_action}'")
num_sequences = input(f"How many sequences to collect? (default: 30): ")
try:
    no_sequences = int(num_sequences) if num_sequences else 30
except:
    no_sequences = 30

# Ask starting sequence number (for appending to existing data)
start_seq = input(f"Start from sequence number? (default: 0): ")
try:
    start_sequence = int(start_seq) if start_seq else 0
except:
    start_sequence = 0

sequence_length = 30

print("\n" + "="*60)
print(f"📋 COLLECTION SETTINGS")
print("="*60)
print(f"Action: {selected_action}")
print(f"Sequences: {no_sequences}")
print(f"Starting from: #{start_sequence}")
print(f"Frames per sequence: {sequence_length}")
print("="*60 + "\n")

# Create directory
os.makedirs(os.path.join(DATA_PATH, selected_action), exist_ok=True)

mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    """
    Extracts coordinates relative to the wrist.
    Total: 258 features (132 pose + 63 left hand + 63 right hand)
    """
    # Pose - 33 landmarks × 4 = 132 features
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] 
                        for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33*4)
        
    # Left Hand - 21 landmarks × 3 = 63 features
    if results.left_hand_landmarks:
        wrist = results.left_hand_landmarks.landmark[0]
        lh = np.array([[res.x - wrist.x, res.y - wrist.y, res.z - wrist.z] 
                      for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21*3)

    # Right Hand - 21 landmarks × 3 = 63 features
    if results.right_hand_landmarks:
        wrist = results.right_hand_landmarks.landmark[0]
        rh = np.array([[res.x - wrist.x, res.y - wrist.y, res.z - wrist.z] 
                      for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21*3)
    
    result = np.concatenate([pose, lh, rh])
    assert result.shape[0] == 258, f"Expected 258 features, got {result.shape[0]}"
    return result

def check_hand_detected(results):
    """Check if at least one hand is detected"""
    return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None

# Camera setup
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Camera 0 not available, trying camera 1...")
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ ERROR: Could not open any camera!")
    exit()

print("\n⚠️  IMPORTANT TIPS FOR '{}':\n".format(selected_action))

# Action-specific instructions
if selected_action == 'Hello':
    print("  🖐️  HELLO SIGN:")
    print("     - Wave your hand side to side (horizontal)")
    print("     - Keep fingers spread open")
    print("     - Perform slowly over 3 seconds")
    print("     - Smile and be friendly!\n")
elif selected_action == 'NO':
    print("  ✋ NO SIGN:")
    print("     - Shake head while moving hand horizontally")
    print("     - OR use index finger shaking side to side")
    print("     - Keep hand in 'stop' gesture")
    print("     - Make it VERY different from Hello\n")
else:
    print(f"  Perform the '{selected_action}' sign consistently\n")

print("  General tips:")
print("  - Hold sign for FULL 3 seconds")
print("  - Ensure good lighting")
print("  - Vary position slightly between sequences")
print("  - Press 'q' to skip a bad sequence")
print("  - Press 'r' to redo last sequence")
print("\nPress SPACE to start!\n")

# Wait for user
while True:
    ret, frame = cap.read()
    if not ret: continue
    
    cv2.putText(frame, f"Ready to collect: {selected_action}", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Sequences: {no_sequences} (starting from #{start_sequence})", (50, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "Press SPACE to start | Q to quit", (50, 130), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow('Data Collection', frame)
    
    key = cv2.waitKey(10)
    if key == ord(' '):
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    sequences_collected = 0
    sequence_num = start_sequence
    
    while sequences_collected < no_sequences:
        window = []
        frames_with_hands = 0
        
        print(f"\n📹 Collecting sequence #{sequence_num} ({sequences_collected + 1}/{no_sequences})")
        
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            if not ret: 
                print("⚠️  Camera read failed!")
                break

            # Process frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw landmarks with distinct colors
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, 
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
            )
            
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, 
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            
            # Count frames with hand detection
            if check_hand_detected(results):
                frames_with_hands += 1
            
            # UI Feedback
            if frame_num == 0: 
                cv2.putText(image, 'GET READY!', (50, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                cv2.putText(image, f'{selected_action} - Sequence #{sequence_num}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow('Data Collection', image)
                cv2.waitKey(2000)
            else:
                # Progress bar
                progress = int((frame_num / sequence_length) * 500)
                cv2.rectangle(image, (10, 450), (510, 470), (50, 50, 50), -1)
                cv2.rectangle(image, (10, 450), (10 + progress, 470), (0, 255, 0), -1)
                
                # Info overlay
                cv2.putText(image, f'{selected_action} - Sequence #{sequence_num}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(image, f'Progress: {sequences_collected + 1}/{no_sequences}', 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(image, f'Frame: {frame_num}/{sequence_length}', 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Hand detection status
                hand_status = "✓ HANDS VISIBLE" if check_hand_detected(results) else "✗ NO HANDS"
                hand_color = (0, 255, 0) if check_hand_detected(results) else (0, 0, 255)
                cv2.putText(image, hand_status, (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)
                
                # Controls hint
                cv2.putText(image, "Q: Skip | R: Redo last", (10, image.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.imshow('Data Collection', image)
            
            # Extract and save keypoints
            keypoints = extract_keypoints(results)
            window.append(keypoints)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'): 
                print("⏭️  Skipping this sequence...")
                break
            elif key == ord('r') and sequences_collected > 0:
                print("🔄 Redoing last sequence...")
                sequences_collected -= 1
                sequence_num -= 1
                break
        
        # Save if complete
        if len(window) == sequence_length:
            hand_detection_rate = frames_with_hands / sequence_length
            
            # Warn if poor quality
            if hand_detection_rate < 0.5:
                print(f"⚠️  WARNING: Only {hand_detection_rate*100:.0f}% frames had hands!")
                print("   Keep? (y/n/r=redo): ", end='', flush=True)
                
                # Show warning on screen
                for _ in range(30):  # Show for 3 seconds
                    ret, frame = cap.read()
                    cv2.putText(frame, "LOW QUALITY DETECTED!", (50, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.putText(frame, "Press Y to keep, N to skip, R to redo", (50, 250), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow('Data Collection', frame)
                    
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('n') or key == ord('N'):
                        print("❌ Skipped")
                        sequence_num += 1
                        break
                    elif key == ord('y') or key == ord('Y'):
                        print("✓ Keeping")
                        break
                    elif key == ord('r') or key == ord('R'):
                        print("🔄 Redoing")
                        break
                else:
                    print("⏱️  Timeout - keeping")
                
                if key == ord('n') or key == ord('N'):
                    continue
                elif key == ord('r') or key == ord('R'):
                    continue
            
            # Save the sequence
            npy_path = os.path.join(DATA_PATH, selected_action, str(sequence_num))
            np.save(npy_path, np.array(window))
            print(f"✅ Saved: {npy_path}.npy (hands: {hand_detection_rate*100:.0f}%)")
            
            sequences_collected += 1
            sequence_num += 1
            
            # Short break
            if sequences_collected < no_sequences:
                time.sleep(0.5)
        else:
            print(f"❌ Incomplete sequence ({len(window)}/{sequence_length} frames)")
            sequence_num += 1

print("\n" + "="*60)
print(f"✅ COLLECTION COMPLETE!")
print("="*60)
print(f"\nCollected {sequences_collected} sequences for '{selected_action}'")
print(f"Saved to: {DATA_PATH}/{selected_action}/")

# Count total sequences now
total_sequences = len([f for f in os.listdir(os.path.join(DATA_PATH, selected_action)) 
                       if f.endswith('.npy')])
print(f"Total sequences for '{selected_action}': {total_sequences}")

print("\n📊 Next steps:")
print(f"  1. If you want more data, run this script again")
print(f"  2. Collect data for other actions if needed")
print(f"  3. Once all actions have 30+ sequences, run train_model.py")
print(f"  4. Recommended: 50+ sequences per action for best accuracy\n")

cap.release()
cv2.destroyAllWindows()