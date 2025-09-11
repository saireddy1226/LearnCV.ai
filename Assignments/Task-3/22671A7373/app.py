import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Image Processing App")
st.write("Upload an image and apply simple operations (based on project work)")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and show original image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    st.image(img_array, caption="Original Image", use_column_width=True)

    # Choose operation
    option = st.selectbox(
        "Select an operation",
        ("Grayscale", "Gaussian Blur", "Edge Detection", "Rotate 90°", "Resize (200x200)", "Flip Horizontal")
    )

    processed_img = None

    if option == "Grayscale":
        processed_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    elif option == "Gaussian Blur":
        processed_img = cv2.GaussianBlur(img_array, (7, 7), 0)

    elif option == "Edge Detection":
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        processed_img = cv2.Canny(gray, 100, 200)

    elif option == "Rotate 90°":
        processed_img = cv2.rotate(img_array, cv2.ROTATE_90_CLOCKWISE)

    elif option == "Resize (200x200)":
        processed_img = cv2.resize(img_array, (200, 200))

    elif option == "Flip Horizontal":
        processed_img = cv2.flip(img_array, 1)

    # Show processed image
    if processed_img is not None:
        st.image(processed_img, caption=f"{option} Image", use_column_width=True)

