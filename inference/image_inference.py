import tensorflow as tf
import cv2
import numpy as np
import sys

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "../models/model_week2.keras"
IMG_SIZE = 224

# -----------------------------
# Load model
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully.")

# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unable to read.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------
# Prediction
# -----------------------------
def predict(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)[0][0]

    if prediction >= 0.5:
        result = "Defect"
    else:
        result = "Pass"

    print(f"Prediction: {result}")
    print(f"Confidence Score: {prediction:.4f}")

# -----------------------------
# Command line interface
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python image_inference.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    predict(image_path)
