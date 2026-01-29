import tensorflow as tf
import cv2
import numpy as np

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "../models/model_week2.keras"
IMG_SIZE = 224
THRESHOLD = 0.5

# -----------------------------
# Load model
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully.")

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
    frame_normalized = frame_resized / 255.0
    frame_expanded = np.expand_dims(frame_normalized, axis=0)
    return frame_expanded

# -----------------------------
# Video Inference
# -----------------------------
def run_video_inference(video_source=0):
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_frame = preprocess_frame(frame)
        prediction = model.predict(input_frame, verbose=0)[0][0]

        if prediction >= THRESHOLD:
            label = "DEFECT"
            color = (0, 0, 255)  # Red
        else:
            label = "PASS"
            color = (0, 255, 0)  # Green

        confidence = f"{prediction:.2f}"

        # Draw results
        cv2.putText(frame, f"Status: {label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Confidence: {confidence}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("VisionSpec QC - Real-Time Inspection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # 0 = webcam
    # Or replace 0 with a video file path like: "../sample_videos/pcb_line.mp4"
    run_video_inference(0)
