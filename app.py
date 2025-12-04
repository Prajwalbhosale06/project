from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import cv2
import pickle
import numpy as np
import base64
from cvzone.HandTrackingModule import HandDetector
import os

# 🔽 TRANSLATION IMPORTS
from googletrans import Translator
from gtts import gTTS
import io

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

print("Loading sign language model...")
try:
    with open('model.p', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: model.p not found.")
    model = None

detector = HandDetector(maxHands=2)
translator = Translator()   # translator instance

# ----------------- SIGN DETECTION HELPERS -----------------

def get_normalized_landmarks(hand):
    lmList = hand['lmList']
    x, y, w, h = hand['bbox']
    normalized = []
    for lm in lmList:
        norm_x = (lm[0] - x) / w
        norm_y = (lm[1] - y) / h
        normalized.extend([norm_x, norm_y])
    return normalized

def process_frame_for_sign_detection(frame_data):
    if model is None:
        return None
    try:
        img_bytes = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return None

        hands, img = detector.findHands(img, draw=False)
        
        if hands:
            data_aux = []
            hand1 = hands[0]
            data_aux.extend(get_normalized_landmarks(hand1))
            
            if len(hands) == 2:
                hand2 = hands[1]
                data_aux.extend(get_normalized_landmarks(hand2))
            else:
                # pad for second hand (21 points * 2 coords = 42)
                data_aux.extend([0] * 42) 
            
            prediction = model.predict([data_aux])
            probabilities = model.predict_proba([data_aux])
            confidence = np.max(probabilities)
            
            if confidence > 0.7:
                return {
                    'sign': prediction[0],
                    'confidence': float(confidence),
                    'hands_detected': len(hands)
                }
        return None
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None

rooms = {}

# ----------------- ROUTES -----------------

@app.route('/')
def index():
    # index.html tumhari templates folder me hona chahiye
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

# 🔽 UPDATED & ROBUST AUDIO TRANSLATION ENDPOINT
@app.route('/translate-audio', methods=['POST'])
def translate_audio():
    try:
        data = request.get_json(force=True) or {}
        text = data.get("text")
        target_lang = data.get("language", "en")  # default English

        print("=== /translate-audio called ===")
        print("Raw data from client:", data)

        if not text:
            print("No text received from client")
            return jsonify({"error": "No text received"}), 400

        print(f"Requested translation: '{text}' -> '{target_lang}'")

        # 1️⃣ Try translation (googletrans)
        try:
            result = translator.translate(text, dest=target_lang)
            translated_text = result.text
            print("googletrans translated text:", translated_text)
        except Exception as e:
            print("googletrans FAILED, using original text. Error:", e)
            # fallback: original text
            translated_text = text

        # 2️⃣ Try TTS (gTTS)
        audio_data_url = None
        try:
            tts = gTTS(translated_text, lang=target_lang)
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)

            audio_b64 = base64.b64encode(audio_bytes.read()).decode("utf-8")
            audio_data_url = f"data:audio/mp3;base64,{audio_b64}"
            print("gTTS audio generated, base64 length:", len(audio_b64))
        except Exception as e:
            print("gTTS FAILED, sending text only. Error:", e)
            # audio_data_url remains None

        return jsonify({
            "translated_text": translated_text,
            "audio_base64": audio_data_url
        })

    except Exception as e:
        print("Translation error (outer exception):", e)
        return jsonify({"error": str(e)}), 500

# ----------------- SOCKET.IO EVENTS -----------------

@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')
    emit('connected', {'sid': request.sid})

@socketio.on('disconnect')
def handle_disconnect():
    print(f'Client disconnected: {request.sid}')
    for room_id in list(rooms.keys()):
        if request.sid in rooms[room_id]:
            rooms[room_id].remove(request.sid)
            if len(rooms[room_id]) == 0:
                del rooms[room_id]
            else:
                emit('user-disconnected', {'userId': request.sid}, room=room_id)

@socketio.on('join-room')
def handle_join_room(data):
    room_id = data.get('roomId')
    user_name = data.get('userName', 'Anonymous')
    if not room_id:
        return
    
    join_room(room_id)
    
    if room_id not in rooms:
        rooms[room_id] = []
    
    rooms[room_id].append(request.sid)
    print(f'User {user_name} ({request.sid}) joined room {room_id}')
    
    emit('user-joined', {
        'userId': request.sid,
        'userName': user_name,
        'roomUsers': rooms[room_id]
    }, room=room_id, include_self=False)
    
    emit('room-users', {'users': rooms[room_id]})

@socketio.on('leave-room')
def handle_leave_room(data):
    room_id = data.get('roomId')
    if room_id and room_id in rooms:
        leave_room(room_id)
        if request.sid in rooms[room_id]:
            rooms[room_id].remove(request.sid)
        if len(rooms[room_id]) == 0:
            del rooms[room_id]
        else:
            emit('user-disconnected', {'userId': request.sid}, room=room_id)

@socketio.on('webrtc-offer')
def handle_offer(data):
    target_id = data.get('targetId')
    offer = data.get('offer')
    emit('webrtc-offer', {'offer': offer, 'senderId': request.sid}, room=target_id)

@socketio.on('webrtc-answer')
def handle_answer(data):
    target_id = data.get('targetId')
    answer = data.get('answer')
    emit('webrtc-answer', {'answer': answer, 'senderId': request.sid}, room=target_id)

@socketio.on('webrtc-ice-candidate')
def handle_ice_candidate(data):
    target_id = data.get('targetId')
    candidate = data.get('candidate')
    emit('webrtc-ice-candidate', {'candidate': candidate, 'senderId': request.sid}, room=target_id)

@socketio.on('process-frame')
def handle_process_frame(data):
    frame_data = data.get('frame')
    room_id = data.get('roomId')
    
    if not frame_data or not room_id:
        return
        
    result = process_frame_for_sign_detection(frame_data)
    
    if result:
        emit('sign-detected', {
            'userId': request.sid,
            'sign': result['sign'],
            'confidence': result['confidence'],
            'timestamp': data.get('timestamp')
        }, room=room_id)

@socketio.on('send-message')
def handle_send_message(data):
    room_id = data.get('roomId')
    message = data.get('message')
    user_name = data.get('userName', 'Anonymous')
    
    if room_id and message:
        emit('receive-message', {
            'userId': request.sid,
            'userName': user_name,
            'message': message,
            'timestamp': data.get('timestamp')
        }, room=room_id)

# ----------------- MAIN -----------------

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"Starting server on port {port}...")
    socketio.run(app, host='0.0.0.0', port=port, debug=True)