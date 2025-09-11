
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# ---------- Helper Functions ----------

def load_image(file):
    image = Image.open(file).convert('RGB')
    return np.array(image)

def convert_color(img, option):
    if option == "RGB ↔ BGR":
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if option == "RGB ↔ HSV":
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if option == "RGB ↔ YCbCr":
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if option == "RGB ↔ Grayscale":
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

def scale_image(img, factor):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * factor), int(h * factor)))

def translate_image(img, tx, ty):
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h))

def affine_transform(img):
    h, w = img.shape[:2]
    pts1 = np.float32([[0,0], [w-1,0], [0,h-1]])
    pts2 = np.float32([[0,h*0.33], [w*0.85,h*0.25], [w*0.15,h*0.7]])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, M, (w, h))

def perspective_transform(img):
    h, w = img.shape[:2]
    pts1 = np.float32([[0,0], [w-1,0], [0,h-1], [w-1,h-1]])
    pts2 = np.float32([[w*0.05,h*0.33], [w*0.9,h*0.1], [w*0.1,h*0.7], [w*0.85,h*0.9]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h))

def apply_filter(img, filter_name, ksize):
    if filter_name == "Gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    if filter_name == "Mean":
        return cv2.blur(img, (ksize, ksize))
    if filter_name == "Median":
        return cv2.medianBlur(img, ksize)
    return img

def apply_morphology(img, morph_name, ksize):
    kernel = np.ones((ksize, ksize), np.uint8)
    if morph_name == "Dilation":
        return cv2.dilate(img, kernel, iterations=1)
    if morph_name == "Erosion":
        return cv2.erode(img, kernel, iterations=1)
    if morph_name == "Opening":
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    if morph_name == "Closing":
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img

def enhance(img, enhancement):
    if enhancement == "Histogram Equalization":
        if len(img.shape) == 2:
            return cv2.equalizeHist(img)
        else:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    if enhancement == "Contrast Stretching":
        min_val = np.min(img)
        max_val = np.max(img)
        stretched = (img - min_val) * 255 / (max_val - min_val)
        return stretched.astype(np.uint8)
    if enhancement == "Sharpening":
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        return cv2.filter2D(img, -1, kernel)
    return img

def edge_detect(img, edge_type, t1, t2):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if edge_type == "Sobel":
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    if edge_type == "Laplacian":
        lap = cv2.Laplacian(img, cv2.CV_64F)
        return cv2.convertScaleAbs(lap)
    if edge_type == "Canny":
        return cv2.Canny(img, t1, t2)
    return img

def compress(img, fmt):
    is_success, buffer = cv2.imencode(fmt, img)
    return buffer

# ---------- Main Streamlit App ----------

st.set_page_config(page_title="Image Processing Toolkit", layout="wide")
st.title("📷 Image Processing Toolkit")

# Sidebar
st.sidebar.header("Options")

uploaded_file = st.sidebar.file_uploader("Open → Upload Image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    original = load_image(uploaded_file)
    processed = original.copy()

    # Image Info
    if st.sidebar.checkbox("Show Image Info"):
        st.sidebar.write(f"Resolution: {original.shape[1]} x {original.shape[0]}")
        st.sidebar.write(f"Channels: {original.shape[2] if len(original.shape) > 2 else 1}")
        st.sidebar.write(f"File Format: {uploaded_file.type}")
        st.sidebar.write(f"File Size: {uploaded_file.size / 1024:.2f} KB")

    # Color Conversion
    color_op = st.sidebar.selectbox("Color Conversions", ["None", "RGB ↔ BGR", "RGB ↔ HSV", "RGB ↔ YCbCr", "RGB ↔ Grayscale"])
    if color_op != "None":
        processed = convert_color(processed, color_op)

    # Transformations
    trans_op = st.sidebar.selectbox("Transformations", ["None", "Rotation", "Scaling", "Translation", "Affine Transform", "Perspective Transform"])
    if trans_op == "Rotation":
        angle = st.sidebar.slider("Rotation Angle", -180, 180, 0)
        processed = rotate_image(processed, angle)
    if trans_op == "Scaling":
        factor = st.sidebar.slider("Scaling Factor", 0.1, 3.0, 1.0)
        processed = scale_image(processed, factor)
    if trans_op == "Translation":
        tx = st.sidebar.slider("Translate X", -100, 100, 0)
        ty = st.sidebar.slider("Translate Y", -100, 100, 0)
        processed = translate_image(processed, tx, ty)
    if trans_op == "Affine Transform":
        processed = affine_transform(processed)
    if trans_op == "Perspective Transform":
        processed = perspective_transform(processed)

    # Filtering
    filter_op = st.sidebar.selectbox("Filtering", ["None", "Gaussian", "Mean", "Median"])
    ksize = st.sidebar.slider("Kernel Size", 3, 15, 3, 2)
    if filter_op != "None":
        processed = apply_filter(processed, filter_op, ksize)

    # Morphology
    morph_op = st.sidebar.selectbox("Morphology", ["None", "Dilation", "Erosion", "Opening", "Closing"])
    if morph_op != "None":
        processed = apply_morphology(processed, morph_op, ksize)

    # Enhancement
    enh_op = st.sidebar.selectbox("Enhancement", ["None", "Histogram Equalization", "Contrast Stretching", "Sharpening"])
    if enh_op != "None":
        processed = enhance(processed, enh_op)

    # Edge Detection
    edge_op = st.sidebar.selectbox("Edge Detection", ["None", "Sobel", "Laplacian", "Canny"])
    if edge_op != "None":
        t1 = st.sidebar.slider("Threshold1", 0, 255, 50)
        t2 = st.sidebar.slider("Threshold2", 0, 255, 150)
        processed = edge_detect(processed, edge_op, t1, t2)

    # Compression
    comp_op = st.sidebar.selectbox("Compression", ["None", "JPG", "PNG", "BMP"])
    if comp_op != "None":
        fmt = '.' + comp_op.lower()
        buffer = compress(processed, fmt)
        size_kb = len(buffer) / 1024
        st.sidebar.write(f"Compressed Size: {size_kb:.2f} KB")

    # Layout
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(original, use_column_width=True)
    with col2:
        st.subheader("Processed Image")
        st.image(processed, use_column_width=True)

    # Save Image
    if st.sidebar.button("Save → Download Processed Image"):
        fmt = st.sidebar.selectbox("Select Format", ["PNG", "JPG", "BMP"])
        img_pil = Image.fromarray(processed)
        buf = io.BytesIO()
        img_pil.save(buf, format=fmt)
        byte_im = buf.getvalue()
        st.sidebar.download_button("Download Image", byte_im, file_name=f"processed.{fmt.lower()}", mime=f"image/{fmt.lower()}")

else:
    st.info("Please upload an image to begin.")

# Status Bar
st.markdown("---")
st.text("Status:")
if uploaded_file is not None:
    h, w = processed.shape[:2]
    c = processed.shape[2] if len(processed.shape) > 2 else 1
    st.text(f"Dimensions: {w} x {h} x {c}")
    st.text(f"File Format: {uploaded_file.type}")
    st.text(f"File Size: {uploaded_file.size / 1024:.2f} KB")
