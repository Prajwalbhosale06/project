from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
import io

# IMPORTS
import mediapipe as mp
from tensorflow.keras.models import load_model
from deep_translator import GoogleTranslator
from gtts import gTTS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# --- 1. GLOBAL MEDIAPIPE SETUP ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Initialize ONE instance to reuse (Much faster)
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# --- 2. LOAD MODEL ---
print("Loading dynamic sign language model...")
try:
    model = load_model('action.h5') 
    # ⚠️ CRITICAL FIX: Must EXACTLY match training labels (case-sensitive!)
    actions = np.array(['Hello', 'NO'])  # Changed 'No' to 'NO' to match training
    print(f"Model loaded successfully! Actions: {actions}")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model. Check action.h5 path. {e}")
    model = None

# Store user history: { 'socket_id': [frame1, frame2, ...] }
user_sequences = {} 

# --- 3. EXTRACTION LOGIC (NORMALIZED) ---
def extract_keypoints(results):
    """
    Extracts coordinates relative to the wrist (Normalization).
    MUST match the logic used in training.
    Total: 258 features (132 pose + 63 left hand + 63 right hand)
    """
    # 1. Pose (Keep raw as anchor)
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] 
                        for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33*4)  # 132 features
        
    # 2. Left Hand (Relative to Wrist)
    if results.left_hand_landmarks:
        wrist = results.left_hand_landmarks.landmark[0]
        lh = np.array([[res.x - wrist.x, res.y - wrist.y, res.z - wrist.z] 
                      for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21*3)  # 63 features

    # 3. Right Hand (Relative to Wrist)
    if results.right_hand_landmarks:
        wrist = results.right_hand_landmarks.landmark[0]
        rh = np.array([[res.x - wrist.x, res.y - wrist.y, res.z - wrist.z] 
                      for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21*3)  # 63 features
    
    result = np.concatenate([pose, lh, rh])
    print(f"[DEBUG] Extracted keypoints shape: {result.shape}")  # Should be (258,)
    return result

def process_frame_for_sign_detection(frame_data, sid):
    """Process frame and return detected sign if confidence > threshold"""
    if model is None: 
        print("[ERROR] Model not loaded!")
        return None

    # Decode Image
    try:
        if ',' in frame_data:
            img_bytes = base64.b64decode(frame_data.split(',')[1])
        else:
            img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: 
            print("[ERROR] Failed to decode image")
            return None
    except Exception as e:
        print(f"[ERROR] Image Decode Error: {e}")
        return None

    # Initialize sequence for new user
    if sid not in user_sequences:
        user_sequences[sid] = []
        print(f"[INFO] Initialized sequence buffer for user: {sid}")

    # --- PROCESS FRAME ---
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(img_rgb)
    
    # Extract Normalized Keypoints
    keypoints = extract_keypoints(results)
    
    # Verify keypoints shape
    if keypoints.shape[0] != 258:
        print(f"[ERROR] Keypoint shape mismatch! Expected 258, got {keypoints.shape[0]}")
        return None
    
    # Add to history
    user_sequences[sid].append(keypoints)
    user_sequences[sid] = user_sequences[sid][-30:]  # Keep last 30 frames

    # PREDICT when we have 30 frames
    if len(user_sequences[sid]) == 30:
        input_data = np.expand_dims(user_sequences[sid], axis=0)
        
        print(f"[DEBUG] Input shape for prediction: {input_data.shape}")  # Should be (1, 30, 258)
        
        # Verbose=0 prevents spamming the terminal
        prediction = model.predict(input_data, verbose=0)[0]
        max_prob_index = np.argmax(prediction)
        confidence = prediction[max_prob_index]
        sign_text = actions[max_prob_index]

        # Enhanced Debug Log
        print(f"[PREDICTION] User {sid[:8]}... -> '{sign_text}' (confidence: {confidence:.3f}) | All probs: {prediction}")

        # ✅ FIXED: Removed 'Idle' check (not in training data)
        # Only require high confidence
        if confidence > 0.80:  # Lowered from 0.8 to catch more predictions
            print(f"[✓ DETECTED] Emitting sign: {sign_text}")
            return {'sign': sign_text, 'confidence': float(confidence)}
        else:
            print(f"[✗ FILTERED] Confidence too low: {confidence:.3f}")

    return None

# ----------------- ROUTES -----------------
@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/translate-audio', methods=['POST'])
def translate_audio():
    try:
        data = request.get_json(force=True) or {}
        text, target_lang = data.get("text"), data.get("language", "en")
        if not text: 
            return jsonify({"error": "No text"}), 400

        # Translate
        try:
            if target_lang != 'en':
                translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
            else:
                translated = text
        except Exception as e:
            print(f"[WARN] Translation failed: {e}")
            translated = text

        # Generate Audio
        audio_url = None
        try:
            tts = gTTS(translated, lang=target_lang)
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            audio_b64 = base64.b64encode(audio_bytes.read()).decode("utf-8")
            audio_url = f"data:audio/mp3;base64,{audio_b64}"
        except Exception as e:
            print(f"[WARN] TTS failed: {e}")

        return jsonify({"translated_text": translated, "audio_base64": audio_url})
    except Exception as e: 
        print(f"[ERROR] translate-audio endpoint: {e}")
        return jsonify({"error": str(e)}), 500

# ----------------- SOCKET EVENTS -----------------
rooms = {}

@socketio.on('connect')
def handle_connect():
    print(f"[SOCKET] User connected: {request.sid}")
    emit('connected', {'sid': request.sid})

@socketio.on('disconnect')
def handle_disconnect():
    print(f"[SOCKET] User disconnected: {request.sid}")
    if request.sid in user_sequences: 
        del user_sequences[request.sid]
    for r in list(rooms.keys()):
        if request.sid in rooms[r]:
            rooms[r].remove(request.sid)
            if len(rooms[r]) == 0: 
                del rooms[r]
            else: 
                emit('user-disconnected', {'userId': request.sid}, room=r)

@socketio.on('join-room')
def handle_join_room(data):
    room_id = data['roomId']
    join_room(room_id)
    if room_id not in rooms: 
        rooms[room_id] = []
    rooms[room_id].append(request.sid)
    print(f"[ROOM] User {request.sid[:8]}... joined room {room_id}")
    emit('user-joined', {
        'userId': request.sid, 
        'userName': data.get('userName'), 
        'roomUsers': rooms[room_id]
    }, room=room_id, include_self=False)

@socketio.on('process-frame')
def handle_process_frame(data):
    """Main frame processing handler"""
    result = process_frame_for_sign_detection(data.get('frame'), request.sid)
    if result:
        print(f"[EMIT] Sending sign-detected event: {result['sign']}")
        emit('sign-detected', {
            'userId': request.sid,
            'userName': data.get('userName'),
            'sign': result['sign'],
            'confidence': result['confidence']
        }, room=data.get('roomId'))

# WebRTC Signaling
@socketio.on('webrtc-offer')
def handle_offer(data):
    data['senderId'] = request.sid
    emit('webrtc-offer', data, room=data.get('targetId'))

@socketio.on('webrtc-answer')
def handle_answer(data):
    data['senderId'] = request.sid
    emit('webrtc-answer', data, room=data.get('targetId'))

@socketio.on('webrtc-ice-candidate')
def handle_ice(data):
    data['senderId'] = request.sid
    emit('webrtc-ice-candidate', data, room=data.get('targetId'))

@socketio.on('send-message')
def handle_msg(data): 
    emit('receive-message', data, room=data.get('roomId'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"\n{'='*50}")
    print(f"🚀 Starting Sign Language Server on port {port}")
    print(f"{'='*50}\n")
    socketio.run(app, host='0.0.0.0', port=port, debug=True)