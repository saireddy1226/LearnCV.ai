import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import io

st.set_page_config(page_title="PIXEL PLAYGROUND", layout="wide")

st.title("PIXEL PLAYGROUND")
# Sidebar categories
st.sidebar.title("Operations Menu")
category = st.sidebar.selectbox(
    "Choose Category",
    ["Image Info", "Color Conversions", "Transformations",
     "Filtering & Morphology", "Enhancement", "Edge Detection", "Compression"]
)

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Layout: two columns for original & processed image
col1, col2 = st.columns(2)

processed = None  # placeholder

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)
    img = np.array(pil_img)

    # Show original image
    with col1:
        st.subheader("Original Image")
        st.image(img, use_container_width=True)

    # -------------------
    # Image Info
    # -------------------
    if category == "Image Info":
        st.subheader("Image Information")
        st.write(f"Resolution: {img.shape[1]} x {img.shape[0]} px")
        st.write(f"Shape: {img.shape}")
        st.write(f"Channels: {img.shape[2] if len(img.shape) == 3 else 1}")
        st.write(f"File format: {uploaded_file.type}")
        st.write(f"File size: {uploaded_file.size} bytes")
        dpi = pil_img.info.get("dpi", "Not available")
        st.write(f"DPI: {dpi}")

    # -------------------
    # Color Conversions
    # -------------------
    elif category == "Color Conversions":
        option = st.sidebar.selectbox("Select Conversion",
                                      ["RGB → BGR", "RGB → HSV", "RGB → YCbCr", "RGB → Grayscale"])
        if option == "RGB → BGR":
            processed = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif option == "RGB → HSV":
            processed = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif option == "RGB → YCbCr":
            processed = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        elif option == "RGB → Grayscale":
            processed = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # -------------------
    # Transformations
    # -------------------
    elif category == "Transformations":
        option = st.sidebar.selectbox("Select Transformation",
                                      ["Rotation", "Scaling", "Translation", "Affine", "Perspective"])
        rows, cols = img.shape[:2]
        if option == "Rotation":
            angle = st.sidebar.slider("Rotation Angle", -180, 180, 90)
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            processed = cv2.warpAffine(img, M, (cols, rows))
        elif option == "Scaling":
            scale = st.sidebar.slider("Scaling Factor", 0.1, 2.0, 0.5)
            processed = cv2.resize(img, None, fx=scale, fy=scale)
        elif option == "Translation":
            tx = st.sidebar.slider("Shift X", -100, 100, 50)
            ty = st.sidebar.slider("Shift Y", -100, 100, 50)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            processed = cv2.warpAffine(img, M, (cols, rows))
        elif option == "Affine":
            pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
            pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
            M = cv2.getAffineTransform(pts1, pts2)
            processed = cv2.warpAffine(img, M, (cols, rows))
        elif option == "Perspective":
            pts1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
            pts2 = np.float32([[10, 100], [200, 50], [100, 250], [220, 220]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            processed = cv2.warpPerspective(img, M, (cols, rows))

    # -------------------
    # Filtering & Morphology
    # -------------------
    elif category == "Filtering & Morphology":
        option = st.sidebar.selectbox("Select Operation",
                                      ["Gaussian Blur", "Mean Blur", "Median Blur",
                                       "Sobel", "Laplacian",
                                       "Dilation", "Erosion", "Opening", "Closing"])
        if option in ["Gaussian Blur", "Mean Blur", "Median Blur"]:
            k = st.sidebar.slider("Kernel Size", 1, 15, 5, step=2)
            if option == "Gaussian Blur":
                processed = cv2.GaussianBlur(img, (k, k), 0)
            elif option == "Mean Blur":
                processed = cv2.blur(img, (k, k))
            elif option == "Median Blur":
                processed = cv2.medianBlur(img, k)
        elif option == "Sobel":
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            processed = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        elif option == "Laplacian":
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            processed = cv2.Laplacian(gray, cv2.CV_64F)
        else:
            kernel = np.ones((5, 5), np.uint8)
            if option == "Dilation":
                processed = cv2.dilate(img, kernel, iterations=1)
            elif option == "Erosion":
                processed = cv2.erode(img, kernel, iterations=1)
            elif option == "Opening":
                processed = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            elif option == "Closing":
                processed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # -------------------
    # Enhancement
    # -------------------
    elif category == "Enhancement":
        option = st.sidebar.selectbox("Select Enhancement",
                                      ["Histogram Equalization", "Contrast Stretching", "Sharpen"])
        if option == "Histogram Equalization":
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            processed = cv2.equalizeHist(gray)
        elif option == "Contrast Stretching":
            min_val = np.min(img)
            max_val = np.max(img)
            processed = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        elif option == "Sharpen":
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            processed = cv2.filter2D(img, -1, kernel)

    # -------------------
    # Edge Detection
    # -------------------
    elif category == "Edge Detection":
        option = st.sidebar.selectbox("Select Method", ["Sobel", "Canny", "Laplacian"])
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if option == "Sobel":
            processed = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        elif option == "Canny":
            t1 = st.sidebar.slider("Threshold1", 0, 255, 100)
            t2 = st.sidebar.slider("Threshold2", 0, 255, 200)
            processed = cv2.Canny(gray, t1, t2)
        elif option == "Laplacian":
            processed = cv2.Laplacian(gray, cv2.CV_64F)

    # -------------------
    # Compression
    # -------------------
    elif category == "Compression":
        option = st.sidebar.selectbox("Save As", ["JPG", "PNG", "BMP"])
        file_name = "output." + option.lower()
        cv2.imwrite(file_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        st.success(f"Image saved as {file_name}")
        st.write(f"File Size: {os.path.getsize(file_name)} bytes")
        processed = img

    # Show processed image
    if processed is not None:
        with col2:
            st.subheader("Processed Image")
            st.image(processed, use_container_width=True)

        # -------------------
        # Save Processed Image (Bottom of Sidebar)
        # -------------------
        st.sidebar.markdown("---")
        st.sidebar.subheader("Save Processed Image")
        save_format = st.sidebar.radio("Select Format", ["JPEG", "JPG", "PNG"])
        if st.sidebar.button("💾 Save Image"):
            pil_processed = Image.fromarray(
                processed if processed.ndim == 3 else cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            )
            buf = io.BytesIO()
            pil_processed.save(buf, format=save_format)
            st.download_button(
                label="Download Processed Image",
                data=buf.getvalue(),
                file_name=f"processed.{save_format.lower()}",
                mime=f"image/{save_format.lower()}"
            )

    # -------------------
    # Status Bar
    # -------------------
    st.markdown("---")
    st.subheader("Status Bar")
    st.write(f"Dimensions: {img.shape}")
    st.write(f"File size: {uploaded_file.size} bytes")
    st.write(f"Format: {uploaded_file.type}")

else:
    st.info("Please upload an image to get started.")
