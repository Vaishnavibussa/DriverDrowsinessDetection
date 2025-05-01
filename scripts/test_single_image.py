import cv2
import numpy as np
from keras.models import load_model
from keras.utils import img_to_array


model = load_model("models/drowsiness_model.keras")
labels = ["Awake", "Drowsy", "Yawn"]


image_path = "test_samples/drowsy_eye.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("❌ Failed to load image:", image_path)
    exit()


image = cv2.resize(image, (64, 64))
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
image = image.astype("float32") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)


prediction = model.predict(image, verbose=0)[0]
predicted_index = np.argmax(prediction)
predicted_label = labels[predicted_index]
confidence = prediction[predicted_index] * 100

print(f" Predicted: {predicted_label} ({confidence:.2f}%)")
print(" Class-wise Probabilities:")
for i, label in enumerate(labels):
    print(f"  {label}: {prediction[i]*100:.2f}%")
