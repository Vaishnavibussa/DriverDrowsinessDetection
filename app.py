from flask import Flask, request, jsonify
import cv2
import numpy as np
import dlib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from scipy.spatial import distance as dist
import time

app = Flask(__name__)

# Load models
model = load_model("scripts/models/drowsiness_model.h5")
predictor = dlib.shape_predictor("C:/Users/bussa/OneDrive/Documents/Desktop/WebDev/miniporj/Driver-Drowsiness-ML/models/shape_predictor_68_face_landmarks.dat")

detector = dlib.get_frontal_face_detector()

EAR_THRESHOLD = 0.2

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def get_eye_points(landmarks, eye_indices):
    return [(landmarks.part(i).x, landmarks.part(i).y) for i in eye_indices]

@app.route('/')
def index():
    return app.send_static_file('index.html')  # Serve frontend here

@app.route('/predict', methods=['POST'])
def predict():
    # Receive image file from frontend
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img_bytes = file.read()
    npimg = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if len(faces) == 0:
        return jsonify({'status': 'No Face Detected', 'confidence': 0})

    # For simplicity, just use first detected face
    face = faces[0]
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    landmarks = predictor(gray, face)

    left_eye = get_eye_points(landmarks, range(36, 42))
    right_eye = get_eye_points(landmarks, range(42, 48))
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    ear = (left_ear + right_ear) / 2.0

    # Crop face for CNN model
    face_crop = frame[y1:y2, x1:x2]
    face_crop = cv2.resize(face_crop, (64, 64))
    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_crop = face_crop.astype("float32") / 255.0
    face_crop = img_to_array(face_crop)
    face_crop = np.expand_dims(face_crop, axis=0)

    prediction = model.predict(face_crop, verbose=0)[0]
    predicted_index = np.argmax(prediction)
    confidence = float(prediction[predicted_index] * 100)
    final_status = "Drowsy" if predicted_index == 0 else "Awake"

    # Use EAR threshold as additional check (optional)
    if ear < EAR_THRESHOLD:
        final_status = "Drowsy"

    return jsonify({'status': final_status, 'confidence': confidence})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
