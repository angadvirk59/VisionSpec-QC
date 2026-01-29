# VisionSpec QC — Visual Quality Control System for PCB Inspection

VisionSpec QC is an **explainable, real-time AI vision system** designed for automated inspection of Printed Circuit Boards (PCBs).
It classifies each PCB image or video frame as **Pass** or **Defect** and highlights the defect location using **Grad-CAM heatmaps**, ensuring both accuracy and interpretability.

This project simulates an industrial assembly-line inspection setup using deep learning and computer vision techniques.

---

## 🔍 Problem Statement

In PCB manufacturing, even small soldering defects can lead to product failure.
Manual inspection is slow, inconsistent, and not scalable. The goal is to build a system that:

* Performs **100% inspection**
* Works in **real time**
* Gives **visual explanations** of detected defects
* Is suitable for **low-latency industrial deployment**

---

## 💡 Solution Overview

VisionSpec QC uses a deep learning pipeline with:

* A **MobileNetV2** backbone for fast inference
* Transfer learning for high accuracy with limited industrial data
* Real-time data augmentation to improve robustness
* Grad-CAM for explainable AI visualization
* OpenCV for video stream processing and real-time overlays

---

## ⚙️ Core Features

* ✅ Pass / Defect classification
* 🔥 Transfer learning with MobileNetV2
* 🎨 Real-time data augmentation
* 🧠 Grad-CAM based defect localization
* 🎥 Video stream inference with FPS measurement
* 📊 Confidence score display
* 🚀 Low-latency inference suitable for production

---

## 🛠️ Tech Stack

* **Deep Learning:** TensorFlow / Keras
* **Computer Vision:** OpenCV
* **Model Architecture:** MobileNetV2
* **Explainability:** Grad-CAM
* **Development:** Google Colab, GitHub

---

## 📁 Repository Structure (Planned)

```
VisionSpec-QC/
│
├── training/          # Model training notebooks
├── explainability/    # Grad-CAM implementation
├── inference/         # Image and video inference scripts
├── realtime_demo/     # FPS measurement & streaming demo
├── assets/            # Sample images, graphs, outputs
├── models/            # Model download instructions
└── outputs/           # Sample inference videos
```

---

## 📦 Dataset & Model Files

Due to size limitations, the dataset and trained model are **not uploaded directly**.

* 📁 Dataset: Stored locally / Google Drive
* 🧠 Model: `model_week2.keras` (download link will be added in `models/README.md`)

---

## 📊 Explainability with Grad-CAM

Grad-CAM visualizations highlight the regions that influenced the model’s decision.
This confirms that the system focuses on **solder joints and components**, not background noise, making it suitable for real-world quality control.

---

## 🎥 Real-Time Inference

The system processes video frames sequentially:

* Each frame is preprocessed
* Classified as Pass or Defect
* Confidence score is displayed
* FPS is calculated to evaluate performance

This simulates an industrial camera pipeline.

---

## 🎯 Internship Relevance

This project demonstrates:

* Production-style ML pipeline design
* Model explainability and trust
* Real-time deployment thinking
* Engineering over pure experimentation

---

## 👨‍💻 Author

**Angad Virk**
Internship Project – VisionSpec QC
(Computer Science, AI / Computer Vision)
