import streamlit as st
import cv2
from ultralytics import YOLO
import torch

# Phase 1: Helmet Detection ONLY
st.title("Month 1 Progress: Real-Time Helmet Detection")

# Hardware Check
device = 0 if torch.cuda.is_available() else "cpu"
if device == 0:
    st.sidebar.success("Using GPU: RTX 4050")
else:
    st.sidebar.warning("Using CPU (Slow) - Check CUDA Drivers")

# Load your model
model = YOLO('best.pt')

# Webcam logic
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Run detection
    results = model.predict(frame, conf=0.5, device=device)
    
    # Draw results
    annotated_frame = results[0].plot()
    
    # Show in app
    frame_placeholder.image(annotated_frame, channels="BGR")

cap.release()
