import streamlit as st
import cv2
import numpy as np
from detect_cells import (
    to_grayscale, denoise, threshold_image, morph_open_close, find_and_filter_contours
)

st.set_page_config(page_title="Cell Detection App", layout="centered")
st.title("🔬 Cell Detection App")

uploaded_file = st.file_uploader("Upload a microscope image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Original Image", use_container_width=True)

    # Processing pipeline
    gray = to_grayscale(image)
    denoised = denoise(gray)
    thresh = threshold_image(denoised)
    morphed = morph_open_close(thresh)
    output, cell_count = find_and_filter_contours(morphed)

    st.image(output, caption=f"Detected Cells (Count: {cell_count})", use_container_width=True)
    st.success(f"Total cells detected: {cell_count}")
