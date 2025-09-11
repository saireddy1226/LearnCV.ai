import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Image Processing GUI", layout="wide")

# --- Top Menu ---
st.title("📷 Image Processing GUI - Streamlit")

menu = st.sidebar.selectbox("Menu", ["Open", "Save", "Exit"])

# --- File Upload ---
if menu == "Open":
    uploaded_file = st.sidebar.file_uploader(
        "Upload an Image", type=["jpg", "jpeg", "png", "bmp"]
    )
    if uploaded_file is not None:
        # Read image
        img = Image.open(uploaded_file)
        img_array = np.array(img)
        orig_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        processed_img = orig_img.copy()

        # Sidebar - Operations
        st.sidebar.subheader("Operations")

        operation = st.sidebar.selectbox(
            "Select Operation",
            [
                "Image Info",
                "Color Conversion",
                "Transformation",
                "Filtering & Morphology",
                "Enhancement",
                "Edge Detection",
                "Compression",
            ],
        )

        # --- Image Info ---
        if operation == "Image Info":
            st.write("### Image Information")
            st.write(f"Resolution: {orig_img.shape[1]} x {orig_img.shape[0]}")
            st.write(
                f"Channels: {orig_img.shape[2] if len(orig_img.shape) == 3 else 1}"
            )
            st.write(f"Format: {uploaded_file.type}")
            st.write(f"File Size: {len(uploaded_file.getvalue()) / 1024:.2f} KB")

        # --- Color Conversions ---
        elif operation == "Color Conversion":
            conversion = st.sidebar.selectbox(
                "Choose Conversion", ["RGB→BGR", "RGB→HSV", "RGB→YCbCr", "RGB→Gray"]
            )
            if conversion == "RGB→BGR":
                processed_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
            elif conversion == "RGB→HSV":
                processed_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2HSV)
            elif conversion == "RGB→YCbCr":
                processed_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2YCrCb)
            elif conversion == "RGB→Gray":
                processed_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)

        # --- Transformations ---
        elif operation == "Transformation":
            trans = st.sidebar.selectbox(
                "Choose Transformation", ["Rotation", "Scaling", "Translation"]
            )
            h, w = orig_img.shape[:2]
            if trans == "Rotation":
                angle = st.sidebar.slider("Angle", -180, 180, 45)
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
                processed_img = cv2.warpAffine(orig_img, M, (w, h))
            elif trans == "Scaling":
                scale = st.sidebar.slider("Scale", 0.5, 2.0, 1.0)
                processed_img = cv2.resize(orig_img, None, fx=scale, fy=scale)
            elif trans == "Translation":
                tx = st.sidebar.slider("Shift X", -100, 100, 20)
                ty = st.sidebar.slider("Shift Y", -100, 100, 20)
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                processed_img = cv2.warpAffine(orig_img, M, (w, h))

        # --- Filtering & Morphology ---
        elif operation == "Filtering & Morphology":
            filt = st.sidebar.selectbox(
                "Choose Filter",
                [
                    "Gaussian",
                    "Median",
                    "Sobel",
                    "Laplacian",
                    "Dilation",
                    "Erosion",
                    "Opening",
                    "Closing",
                ],
            )
            if filt == "Gaussian":
                processed_img = cv2.GaussianBlur(orig_img, (7, 7), 0)
            elif filt == "Median":
                processed_img = cv2.medianBlur(orig_img, 5)
            elif filt == "Sobel":
                processed_img = cv2.Sobel(orig_img, cv2.CV_64F, 1, 1, ksize=5)
            elif filt == "Laplacian":
                processed_img = cv2.Laplacian(orig_img, cv2.CV_64F)
            elif filt == "Dilation":
                kernel = np.ones((5, 5), np.uint8)
                processed_img = cv2.dilate(orig_img, kernel, iterations=1)
            elif filt == "Erosion":
                kernel = np.ones((5, 5), np.uint8)
                processed_img = cv2.erode(orig_img, kernel, iterations=1)
            elif filt == "Opening":
                kernel = np.ones((5, 5), np.uint8)
                processed_img = cv2.morphologyEx(orig_img, cv2.MORPH_OPEN, kernel)
            elif filt == "Closing":
                kernel = np.ones((5, 5), np.uint8)
                processed_img = cv2.morphologyEx(orig_img, cv2.MORPH_CLOSE, kernel)

        # --- Enhancement ---
        elif operation == "Enhancement":
            enh = st.sidebar.selectbox(
                "Choose Enhancement",
                ["Histogram Equalization", "Contrast Stretching", "Sharpening"],
            )
            if enh == "Histogram Equalization":
                gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
                processed_img = cv2.equalizeHist(gray)
            elif enh == "Contrast Stretching":
                processed_img = cv2.normalize(orig_img, None, 0, 255, cv2.NORM_MINMAX)
            elif enh == "Sharpening":
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                processed_img = cv2.filter2D(orig_img, -1, kernel)

        # --- Edge Detection ---
        elif operation == "Edge Detection":
            edge = st.sidebar.selectbox(
                "Choose Edge Detector", ["Sobel", "Canny", "Laplacian"]
            )
            if edge == "Sobel":
                processed_img = cv2.Sobel(orig_img, cv2.CV_64F, 1, 0, ksize=5)
            elif edge == "Canny":
                processed_img = cv2.Canny(orig_img, 100, 200)
            elif edge == "Laplacian":
                processed_img = cv2.Laplacian(orig_img, cv2.CV_64F)

        # --- Compression ---
        elif operation == "Compression":
            fmt = st.sidebar.selectbox("Save Format", ["jpg", "png", "bmp"])
            output_path = f"compressed_output.{fmt}"
            cv2.imwrite(output_path, orig_img)
            size = os.path.getsize(output_path) / 1024
            st.write(f"Saved as {fmt.upper()} | File Size: {size:.2f} KB")

        # --- Display ---
        col1, col2 = st.columns(2)
        with col1:
            st.image(
                cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB),
                caption="Original Image",
                use_column_width=True,
            )
        with col2:
            try:
                st.image(
                    cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB),
                    caption="Processed Image",
                    use_column_width=True,
                )
            except:
                st.image(
                    processed_img, caption="Processed Image", use_column_width=True
                )

        # --- Status Bar ---
        st.markdown("---")
        st.write(f"**Dimensions:** {orig_img.shape}")
        st.write(f"**File Size:** {len(uploaded_file.getvalue()) / 1024:.2f} KB")
        st.write(f"**Format:** {uploaded_file.type}")

elif menu == "Exit":
    st.stop()
