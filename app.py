from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import cv2
import pickle
import numpy as np
import base64
from cvzone.HandTrackingModule import HandDetector
import os

# IMPORTS
from deep_translator import GoogleTranslator
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
    if model is None: return None
    try:
        if ',' in frame_data:
            img_bytes = base64.b64decode(frame_data.split(',')[1])
        else:
            img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return None

        hands, img = detector.findHands(img, draw=False)
        if hands:
            data_aux = []
            hand1 = hands[0]
            data_aux.extend(get_normalized_landmarks(hand1))
            if len(hands) == 2:
                hand2 = hands[1]
                data_aux.extend(get_normalized_landmarks(hand2))
            else:
                data_aux.extend([0] * 42) 
            
            prediction = model.predict([data_aux])
            probabilities = model.predict_proba([data_aux])
            confidence = np.max(probabilities)
            
            if confidence > 0.7:
                return {'sign': str(prediction[0]), 'confidence': float(confidence)}
        return None
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None

# ----------------- ROUTES -----------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/translate-audio', methods=['POST'])
def translate_audio():
    """
    This endpoint is called by the RECEIVER'S client.
    It takes text + target_language and returns translated text + audio.
    """
    try:
        data = request.get_json(force=True) or {}
        text = data.get("text")
        target_lang = data.get("language", "en")

        if not text: return jsonify({"error": "No text"}), 400

        # 1. Translate
        try:
            if target_lang != 'en':
                translated_text = GoogleTranslator(source='auto', target=target_lang).translate(text)
            else:
                translated_text = text
        except Exception as e:
            print(f"Translation Error: {e}")
            translated_text = text

        # 2. Generate Audio
        audio_data_url = None
        try:
            tts = gTTS(translated_text, lang=target_lang)
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            audio_b64 = base64.b64encode(audio_bytes.read()).decode("utf-8")
            audio_data_url = f"data:audio/mp3;base64,{audio_b64}"
        except Exception as e:
            print(f"TTS Error: {e}")

        return jsonify({
            "translated_text": translated_text,
            "audio_base64": audio_data_url
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------- SOCKET EVENTS -----------------

rooms = {}

@socketio.on('connect')
def handle_connect():
    emit('connected', {'sid': request.sid})

@socketio.on('disconnect')
def handle_disconnect():
    for room_id in list(rooms.keys()):
        if request.sid in rooms[room_id]:
            rooms[room_id].remove(request.sid)
            if len(rooms[room_id]) == 0: del rooms[room_id]
            else: emit('user-disconnected', {'userId': request.sid}, room=room_id)

@socketio.on('join-room')
def handle_join_room(data):
    room_id, user_name = data.get('roomId'), data.get('userName', 'Anonymous')
    if not room_id: return
    join_room(room_id)
    if room_id not in rooms: rooms[room_id] = []
    rooms[room_id].append(request.sid)
    emit('user-joined', {'userId': request.sid, 'userName': user_name, 'roomUsers': rooms[room_id]}, room=room_id, include_self=False)

@socketio.on('process-frame')
def handle_process_frame(data):
    frame_data = data.get('frame')
    room_id = data.get('roomId')
    user_name = data.get('userName', 'Unknown')
    
    if not frame_data or not room_id: return
        
    result = process_frame_for_sign_detection(frame_data)
    
    if result:
        detected_sign = result['sign']
        emit('sign-detected', {
            'userId': request.sid,
            'userName': user_name,
            'sign': detected_sign,
            'timestamp': data.get('timestamp')
        }, room=room_id)

# Signaling for WebRTC - UPDATED TO INCLUDE SENDER ID
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
def handle_msg(data): emit('receive-message', data, room=data.get('roomId'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"Starting server on port {port}...")
    socketio.run(app, host='0.0.0.0', port=port, debug=True)