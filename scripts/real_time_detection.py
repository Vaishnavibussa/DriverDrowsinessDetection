import os
import cv2
import dlib
import numpy as np
import winsound
from scipy.spatial import distance as dist
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model = load_model("models/drowsiness_model.h5")

labels = ["Awake", "Drowsy", "Yawn"]

EAR_THRESHOLD = 0.25

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def get_eye_points(landmarks, eye_indices):
    return [(landmarks.part(i).x, landmarks.part(i).y) for i in eye_indices]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Webcam frame not captured.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        cv2.putText(frame, "No Face Detected", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        for face in faces:
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()


            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            landmarks = predictor(gray, face)
            left_eye = get_eye_points(landmarks, range(36, 42))
            right_eye = get_eye_points(landmarks, range(42, 48))

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            face_crop = frame[y1:y2, x1:x2]
            face_crop = cv2.resize(face_crop, (64, 64))
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_crop = face_crop.astype("float32") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            prediction = model.predict(face_crop, verbose=0)[0]
            predicted_index = np.argmax(prediction)
            predicted_label = labels[predicted_index]
            confidence = prediction[predicted_index] * 100

            if predicted_label == "Awake" and ear > EAR_THRESHOLD:
                final_status = "Awake"
            else:
                final_status = "Drowsy"

            cv2.putText(frame, f"EAR: {ear:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Status: {final_status} ({confidence:.1f}%)", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if final_status == "Awake" else (0, 0, 255), 3)

            if final_status == "Drowsy":
                winsound.Beep(2500, 1000)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
