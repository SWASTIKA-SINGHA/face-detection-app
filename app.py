import streamlit as st

st.set_page_config(page_title="🌸 Face Detector App", page_icon="📷", layout="centered")

st.title("🌸 Face Detector")
st.markdown("Detect faces in your image with a clean and elegant interface 💫")


import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load Haar Cascade model
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

st.title("📸 Face Detection App")
st.write("Upload an image and the app will detect faces!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 2)

    st.image(img_np, caption=f"Detected {len(faces)} face(s)", use_column_width=True)
