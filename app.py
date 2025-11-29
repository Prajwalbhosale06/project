import cv2
import av
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# --- 1. LOAD MODELS ONCE (Global) ---
# We load these outside the class so we don't reload them every frame
detector = HandDetector(maxHands=1)
classifier = Classifier("keras_model.h5", "labels.txt")
labels = ["Hello","I love you","No","Okay","Please","Thank you","Yes"]
offset = 20
imgSize = 300

# --- 2. THE PROCESSOR CLASS ---
class GestureProcessor(VideoTransformerBase):
    def transform(self, frame):
        # 1. Convert WebRTC frame to OpenCV format
        img = frame.to_ndarray(format="bgr24")
        
        # 2. YOUR ORIGINAL LOGIC STARTS HERE
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        
        if hands:
            try:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
                imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
                
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize-wCal)/2)
                    imgWhite[:, wGap: wCal + wGap] = imgResize
                    prediction , index = classifier.getPrediction(imgWhite, draw=False)
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap: hCal + hGap, :] = imgResize
                    prediction , index = classifier.getPrediction(imgWhite, draw=False)

                # Draw Results
                cv2.rectangle(imgOutput,(x-offset,y-offset-70),(x -offset+400, y - offset+60-50),(0,255,0),cv2.FILLED)  
                cv2.putText(imgOutput,labels[index],(x,y-30),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),2) 
                cv2.rectangle(imgOutput,(x-offset,y-offset),(x + w + offset, y+h + offset),(0,255,0),4)
                
            except Exception as e:
                pass # Prevent crash if hand goes out of bounds

        # 3. Return the processed frame back to the browser
        return av.VideoFrame.from_ndarray(imgOutput, format="bgr24")

# --- 3. THE UI SETUP ---
webrtc_streamer(key="example", video_transformer_factory=GestureProcessor)