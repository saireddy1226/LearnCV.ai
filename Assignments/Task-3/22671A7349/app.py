import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os

# -------- Utility Functions --------
def load_image(image_file):
    image = Image.open(image_file)
    image = image.convert('RGB')
    return np.array(image)

def convert_color(img, conversion_type):
    if conversion_type == "RGB ↔ BGR":
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif conversion_type == "RGB ↔ HSV":
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif conversion_type == "RGB ↔ YCbCr":
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    elif conversion_type == "RGB ↔ Grayscale":
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        return img

def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

def scale_image(img, scale_factor):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w*scale_factor), int(h*scale_factor)))

def translate_image(img, tx, ty):
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h))

def apply_affine(img):
    h, w = img.shape[:2]
    pts1 = np.float32([[0,0], [w-1,0], [0,h-1]])
    pts2 = np.float32([[0,h*0.33], [w*0.85,h*0.25], [w*0.15,h*0.7]])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, M, (w, h))

def apply_perspective(img):
    h, w = img.shape[:2]
    pts1 = np.float32([[0,0], [w-1,0], [0,h-1], [w-1,h-1]])
    pts2 = np.float32([[w*0.05,h*0.33], [w*0.9,h*0.1], [w*0.1,h*0.7], [w*0.85,h*0.9]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h))

def apply_filter(img, filter_type, ksize=3):
    if filter_type == "Gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    elif filter_type == "Mean":
        return cv2.blur(img, (ksize, ksize))
    elif filter_type == "Median":
        return cv2.medianBlur(img, ksize)
    else:
        return img

def apply_morphology(img, morph_type, ksize=3):
    kernel = np.ones((ksize, ksize), np.uint8)
    if morph_type == "Dilation":
        return cv2.dilate(img, kernel, iterations=1)
    elif morph_type == "Erosion":
        return cv2.erode(img, kernel, iterations=1)
    elif morph_type == "Opening":
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif morph_type == "Closing":
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    else:
        return img

def enhance_image(img, enhancement_type):
    if enhancement_type == "Histogram Equalization":
        if len(img.shape) == 2:
            return cv2.equalizeHist(img)
        else:
            img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            return cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2RGB)
    elif enhancement_type == "Contrast Stretching":
        min_val = np.min(img)
        max_val = np.max(img)
        stretched = (img - min_val) * 255 / (max_val - min_val)
        return stretched.astype(np.uint8)
    elif enhancement_type == "Sharpening":
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        return cv2.filter2D(img, -1, kernel)
    else:
        return img

def edge_detection(img, edge_type, threshold1=50, threshold2=150):
    if edge_type == "Sobel":
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    elif edge_type == "Laplacian":
        lap = cv2.Laplacian(img, cv2.CV_64F)
        return cv2.convertScaleAbs(lap)
    elif edge_type == "Canny":
        return cv2.Canny(img, threshold1, threshold2)
    else:
        return img

def compress_image(img, fmt):
    is_success, buffer = cv2.imencode(fmt, img)
    return buffer

# -------- Streamlit App --------

st.set_page_config(page_title="Image Processing Toolkit", layout="wide")

st.title("🖼 Image Processing Toolkit")

# Sidebar for operations
st.sidebar.title("Operations")

uploaded_file = st.sidebar.file_uploader("Open → Upload an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    original_img = load_image(uploaded_file)
    processed_img = original_img.copy()

    # Image Info
    if st.sidebar.checkbox("Image Info"):
        st.sidebar.text(f"Resolution: {original_img.shape[1]} x {original_img.shape[0]}")
        st.sidebar.text(f"Channels: {original_img.shape[2] if len(original_img.shape) > 2 else 1}")
        st.sidebar.text(f"Format: {uploaded_file.type}")
        st.sidebar.text(f"File Size: {uploaded_file.size/1024:.2f} KB")

    # Color Conversions
    color_ops = st.sidebar.selectbox("Color Conversions", ["None", "RGB ↔ BGR", "RGB ↔ HSV", "RGB ↔ YCbCr", "RGB ↔ Grayscale"])
    if color_ops != "None":
        processed_img = convert_color(processed_img, color_ops)

    # Transformations
    trans_ops = st.sidebar.selectbox("Transformations", ["None", "Rotation", "Scaling", "Translation", "Affine Transform", "Perspective Transform"])
    if trans_ops == "Rotation":
        angle = st.sidebar.slider("Angle", -180, 180, 0)
        processed_img = rotate_image(processed_img, angle)
    elif trans_ops == "Scaling":
        scale = st.sidebar.slider("Scale Factor", 0.1, 3.0, 1.0)
        processed_img = scale_image(processed_img, scale)
    elif trans_ops == "Translation":
        tx = st.sidebar.slider("Shift X", -100, 100, 0)
        ty = st.sidebar.slider("Shift Y", -100, 100, 0)
        processed_img = translate_image(processed_img, tx, ty)
    elif trans_ops == "Affine Transform":
        processed_img = apply_affine(processed_img)
    elif trans_ops == "Perspective Transform":
        processed_img = apply_perspective(processed_img)

    # Filtering & Morphology
    filter_ops = st.sidebar.selectbox("Filtering", ["None", "Gaussian", "Mean", "Median"])
    morph_ops = st.sidebar.selectbox("Morphology", ["None", "Dilation", "Erosion", "Opening", "Closing"])
    ksize = st.sidebar.slider("Kernel Size", 3, 15, 3, 2)

    if filter_ops != "None":
        processed_img = apply_filter(processed_img, filter_ops, ksize)
    if morph_ops != "None":
        processed_img = apply_morphology(processed_img, morph_ops, ksize)

    # Enhancement
    enh_ops = st.sidebar.selectbox("Enhancement", ["None", "Histogram Equalization", "Contrast Stretching", "Sharpening"])
    if enh_ops != "None":
        processed_img = enhance_image(processed_img, enh_ops)

    # Edge Detection
    edge_ops = st.sidebar.selectbox("Edge Detection", ["None", "Sobel", "Laplacian", "Canny"])
    if edge_ops != "None":
        threshold1 = st.sidebar.slider("Threshold1", 0, 255, 50)
        threshold2 = st.sidebar.slider("Threshold2", 0, 255, 150)
        if len(processed_img.shape) == 3:
            gray = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = processed_img
        processed_img = edge_detection(gray, edge_ops, threshold1, threshold2)

    # Compression
    comp_ops = st.sidebar.selectbox("Compression", ["None", "JPG", "PNG", "BMP"])
    if comp_ops != "None":
        fmt = "." + comp_ops.lower()
        buffer = compress_image(processed_img, fmt)
        file_size = len(buffer) / 1024
        st.sidebar.text(f"Compressed Size: {file_size:.2f} KB")

    # Display Area
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(original_img, use_column_width=True)
    with col2:
        st.subheader("Processed Image")
        st.image(processed_img, use_column_width=True)

    # Save Button
    if st.sidebar.button("Save → Download Processed Image"):
        save_format = st.sidebar.selectbox("Save Format", ["PNG", "JPG", "BMP"])
        img_pil = Image.fromarray(processed_img)
        buf = io.BytesIO()
        img_pil.save(buf, format=save_format)
        byte_im = buf.getvalue()
        st.sidebar.download_button("Download Image", byte_im, file_name=f"processed_image.{save_format.lower()}", mime="image/"+save_format.lower())

else:
    st.info("Upload an image file to get started.")

# Status Bar
st.markdown("---")
st.text("Status Bar:")
if uploaded_file is not None:
    st.text(f"Dimensions: {processed_img.shape[1]} x {processed_img.shape[0]} x {processed_img.shape[2] if len(processed_img.shape) > 2 else 1}")
    st.text(f"File Format: {uploaded_file.type}")
    st.text(f"File Size: {uploaded_file.size/1024:.2f} KB")
