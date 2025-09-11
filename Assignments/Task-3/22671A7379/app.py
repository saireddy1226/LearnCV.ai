import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os

st.set_page_config(layout="wide")
st.title("Image Processing & Analysis Toolkit")

# --- Menu Bar (Sidebar as workaround) ---
with st.sidebar:
    st.header("File")
    uploaded_file = st.file_uploader("Open: Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    save_btn = st.button("Save Processed Image")
    exit_btn = st.button("Exit App")

# --- Load Image ---
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    orig_pil = Image.open(io.BytesIO(uploaded_file.getvalue()))
    file_format = orig_pil.format
    file_size = uploaded_file.size
    dpi = orig_pil.info.get("dpi", (72, 72))
    shape = image.shape

    # --- Sidebar: Operations ---
    st.sidebar.header("Operations")
    op_category = st.sidebar.selectbox(
        "Select Category",
        ("Image Info", "Color Conversions", "Transformations", "Filtering & Morphology", "Enhancement", "Edge Detection", "Compression")
    )

    operation = None
    params = {}

    if op_category == "Image Info":
        operation = "info"
    elif op_category == "Color Conversions":
        operation = st.sidebar.selectbox("Conversion", [
            "RGB ↔ BGR", "RGB ↔ HSV", "RGB ↔ YCbCr", "RGB ↔ Grayscale"
        ])
    elif op_category == "Transformations":
        operation = st.sidebar.selectbox("Transformation", [
            "Rotation", "Scaling", "Translation", "Affine Transform", "Perspective Transform"
        ])
        if operation == "Rotation":
            params["angle"] = st.sidebar.slider("Angle", -180, 180, 0)
        elif operation == "Scaling":
            params["scale"] = st.sidebar.slider("Scale", 0.1, 3.0, 1.0, 0.1)
        elif operation == "Translation":
            params["x"] = st.sidebar.slider("Shift X", -200, 200, 0)
            params["y"] = st.sidebar.slider("Shift Y", -200, 200, 0)
    elif op_category == "Filtering & Morphology":
        operation = st.sidebar.selectbox("Filter/Morph", [
            "Gaussian", "Mean", "Median", "Sobel", "Laplacian", "Dilation", "Erosion", "Opening", "Closing"
        ])
        if operation in ["Gaussian", "Mean", "Median"]:
            params["ksize"] = st.sidebar.slider("Kernel Size", 1, 25, 5, step=2)
    elif op_category == "Enhancement":
        operation = st.sidebar.selectbox("Enhancement", [
            "Histogram Equalization", "Contrast Stretching", "Sharpening"
        ])
    elif op_category == "Edge Detection":
        operation = st.sidebar.selectbox("Edge", [
            "Sobel", "Canny", "Laplacian"
        ])
        if operation == "Canny":
            params["low"] = st.sidebar.slider("Canny Low", 0, 255, 100)
            params["high"] = st.sidebar.slider("Canny High", 0, 255, 200)
    elif op_category == "Compression":
        operation = st.sidebar.selectbox("Compression", [
            "Save as JPG", "Save as PNG", "Save as BMP"
        ])

    # --- Processing ---
    processed = image.copy()
    info_text = ""
    if operation == "info":
        info_text = f"""
        **Resolution:** {shape[1]} x {shape[0]}  
        **Channels:** {shape[2] if len(shape) == 3 else 1}  
        **DPI:** {dpi}  
        **Format:** {file_format}  
        **File Size:** {file_size/1024:.2f} KB
        """
    elif operation == "RGB ↔ BGR":
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    elif operation == "RGB ↔ HSV":
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
    elif operation == "RGB ↔ YCbCr":
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2YCrCb)
    elif operation == "RGB ↔ Grayscale":
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    elif operation == "Rotation":
        (h, w) = processed.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), params["angle"], 1.0)
        processed = cv2.warpAffine(processed, M, (w, h))
    elif operation == "Scaling":
        processed = cv2.resize(processed, None, fx=params["scale"], fy=params["scale"])
    elif operation == "Translation":
        M = np.float32([[1, 0, params["x"]], [0, 1, params["y"]]])
        (h, w) = processed.shape[:2]
        processed = cv2.warpAffine(processed, M, (w, h))
    elif operation == "Gaussian":
        processed = cv2.GaussianBlur(processed, (params["ksize"], params["ksize"]), 0)
    elif operation == "Mean":
        processed = cv2.blur(processed, (params["ksize"], params["ksize"]))
    elif operation == "Median":
        processed = cv2.medianBlur(processed, params["ksize"])
    elif operation == "Sobel":
        processed = cv2.Sobel(processed, cv2.CV_64F, 1, 0, ksize=5)
        processed = cv2.convertScaleAbs(processed)
    elif operation == "Laplacian":
        processed = cv2.Laplacian(processed, cv2.CV_64F)
        processed = cv2.convertScaleAbs(processed)
    elif operation == "Dilation":
        kernel = np.ones((5,5), np.uint8)
        processed = cv2.dilate(processed, kernel, iterations=1)
    elif operation == "Erosion":
        kernel = np.ones((5,5), np.uint8)
        processed = cv2.erode(processed, kernel, iterations=1)
    elif operation == "Opening":
        kernel = np.ones((5,5), np.uint8)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
    elif operation == "Closing":
        kernel = np.ones((5,5), np.uint8)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    elif operation == "Histogram Equalization":
        if len(processed.shape) == 2:
            processed = cv2.equalizeHist(processed)
        else:
            ycrcb = cv2.cvtColor(processed, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            processed = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    elif operation == "Contrast Stretching":
        in_min = np.percentile(processed, 2)
        in_max = np.percentile(processed, 98)
        processed = np.clip((processed - in_min) * 255.0 / (in_max - in_min), 0, 255).astype(np.uint8)
    elif operation == "Sharpening":
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        processed = cv2.filter2D(processed, -1, kernel)
    elif operation == "Canny":
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        processed = cv2.Canny(gray, params["low"], params["high"])
    elif operation == "Save as JPG":
        _, buf = cv2.imencode('.jpg', processed)
        st.sidebar.download_button("Download JPG", buf.tobytes(), file_name="processed.jpg", mime="image/jpeg")
    elif operation == "Save as PNG":
        _, buf = cv2.imencode('.png', processed)
        st.sidebar.download_button("Download PNG", buf.tobytes(), file_name="processed.png", mime="image/png")
    elif operation == "Save as BMP":
        _, buf = cv2.imencode('.bmp', processed)
        st.sidebar.download_button("Download BMP", buf.tobytes(), file_name="processed.bmp", mime="image/bmp")

    # --- Display Area (Dual Panel) ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
    with col2:
        st.subheader("Processed Image")
        if operation == "info":
            st.info(info_text)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
        elif operation and len(processed.shape) == 2:
            st.image(processed, use_column_width=True, channels="GRAY")
        elif operation:
            st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_column_width=True)
        else:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

    # --- Status Bar (Bottom) ---
    st.markdown("---")
    st.markdown(
        f"**Dimensions:** {shape[1]} x {shape[0]} &nbsp;&nbsp; "
        f"**Channels:** {shape[2] if len(shape) == 3 else 1} &nbsp;&nbsp; "
        f"**DPI:** {dpi} &nbsp;&nbsp; "
        f"**Format:** {file_format} &nbsp;&nbsp; "
        f"**File Size:** {file_size/1024:.2f} KB"
    )

    # --- Exit Button ---
    if exit_btn:
        st.stop()
else:
    st.info("Please upload an image to begin.")