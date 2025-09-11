
import io
import cv2
import time
import numpy as np
import streamlit as st
from typing import Tuple
from utils import (
    get_image_info, bgr_to_rgb, rgb_to_bgr, rgb_to_hsv, hsv_to_rgb, rgb_to_ycrcb, ycrcb_to_rgb,
    rgb_to_gray, gray_to_rgb, rotate_image, scale_image, translate_image, affine_transform,
    perspective_transform, bitwise_and, bitwise_or, bitwise_xor, bitwise_not, mean_filter,
    gaussian_filter, median_filter, sobel_edges, laplacian_edges, canny_edges, morphology,
    histogram_equalization, contrast_stretch, sharpen, ensure_rgb, ensure_gray, to_3channel,
    split_screen_compare, encode_format
)

st.set_page_config(page_title="Image Processing Toolkit", layout="wide")
st.title("🖼️ Image Processing Toolkit — OpenCV + Streamlit")

# --- Menu (Top) ---
with st.sidebar:
    st.header("📁 File")
    uploaded = st.file_uploader("Open image", type=["png","jpg","jpeg","bmp"])
    save_format = st.selectbox("Save format", ["png","jpg","bmp"], index=0)
    save_btn = st.button("💾 Save Processed Image")
    st.markdown("---")
    st.header("⚙️ Operations")

# Load image
src_bytes = None
orig_rgb = None
if uploaded is not None:
    src_bytes = uploaded.getvalue()
    file_bytes = np.frombuffer(src_bytes, np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if img_bgr is not None and len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    if img_bgr is not None:
        orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Sidebar options (after load)
mode = st.sidebar.selectbox(
    "Choose category",
    ["Image Info", "Color Conversions", "Transformations", "Filtering & Morphology",
     "Enhancement", "Edge Detection", "Compression", "Bitwise Ops", "Video (Bonus)"],
    index=0
)

# --- Display area ---
col1, col2 = st.columns(2, vertical_alignment="center")
with col1:
    st.subheader("Original")
    if orig_rgb is not None:
        st.image(orig_rgb, channels="RGB", use_container_width=True, caption="Original Image")
    else:
        st.info("Upload an image to begin.")

processed = None

# --- Operations ---
if orig_rgb is not None:
    if mode == "Image Info":
        info = get_image_info(cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR), src_bytes, uploaded.name.split(".")[-1] if uploaded else None)
        st.sidebar.json(info)
        processed = orig_rgb.copy()

    elif mode == "Color Conversions":
        conv = st.sidebar.selectbox("Conversion", ["RGB→HSV","HSV→RGB","RGB→YCbCr","YCbCr→RGB","RGB→Gray","Gray→RGB"])
        img = orig_rgb.copy()
        if conv == "RGB→HSV":
            processed = rgb_to_hsv(img)
        elif conv == "HSV→RGB":
            processed = hsv_to_rgb(rgb_to_hsv(img))
        elif conv == "RGB→YCbCr":
            processed = rgb_to_ycrcb(img)
        elif conv == "YCbCr→RGB":
            processed = ycrcb_to_rgb(rgb_to_ycrcb(img))
        elif conv == "RGB→Gray":
            processed = rgb_to_gray(img)
        elif conv == "Gray→RGB":
            processed = gray_to_rgb(rgb_to_gray(img))

    elif mode == "Transformations":
        tmode = st.sidebar.selectbox("Transform", ["Rotation","Scaling","Translation","Affine","Perspective"])
        img = orig_rgb.copy()
        h, w = img.shape[:2]
        if tmode == "Rotation":
            ang = st.sidebar.slider("Angle (deg)", -180, 180, 30)
            processed = rotate_image(img, ang)
        elif tmode == "Scaling":
            fx = st.sidebar.slider("Scale X", 10, 300, 150) / 100.0
            fy = st.sidebar.slider("Scale Y", 10, 300, 150) / 100.0
            processed = scale_image(img, fx, fy)
        elif tmode == "Translation":
            tx = st.sidebar.slider("Shift X", -w//2, w//2, 50)
            ty = st.sidebar.slider("Shift Y", -h//2, h//2, 50)
            processed = translate_image(img, tx, ty)
        elif tmode == "Affine":
            st.sidebar.info("Using triangle corners for demo.")
            src = np.float32([[0,0],[w-1,0],[0,h-1]])
            dst = np.float32([[0,int(0.1*h)],[w-1,0],[int(0.2*w),h-1]])
            processed = affine_transform(img, src, dst)
        elif tmode == "Perspective":
            st.sidebar.info("Warp corners inward for demo.")
            src = np.float32([[0,0],[w-1,0],[0,h-1],[w-1,h-1]])
            dst = np.float32([[int(0.1*w),int(0.1*h)],[int(0.9*w),int(0.05*h)],[int(0.05*w),int(0.9*h)],[int(0.95*w),int(0.95*h)]])
            processed = perspective_transform(img, src, dst)

    elif mode == "Filtering & Morphology":
        fmode = st.sidebar.selectbox("Filter", ["Mean","Gaussian","Median","Sobel","Laplacian","Dilation","Erosion","Opening","Closing"])
        img = orig_rgb.copy()
        if fmode in ["Mean","Gaussian","Median"]:
            k = st.sidebar.slider("Kernel size", 3, 31, 5, step=2)
            if fmode == "Mean":
                processed = mean_filter(img, k)
            elif fmode == "Gaussian":
                processed = gaussian_filter(img, k, 0)
            elif fmode == "Median":
                processed = median_filter(img, k)
        elif fmode in ["Sobel","Laplacian"]:
            gray = ensure_gray(img)
            if fmode == "Sobel":
                processed = sobel_edges(gray)
            else:
                processed = laplacian_edges(gray)
        else:
            k = st.sidebar.slider("Kernel size", 3, 31, 5, step=2)
            it = st.sidebar.slider("Iterations", 1, 5, 1)
            gray_or_rgb = ensure_gray(img)
            op = {"Dilation":"dilate","Erosion":"erode","Opening":"open","Closing":"close"}[fmode]
            morphed = morphology(gray_or_rgb, op, k, it)
            processed = morphed

    elif mode == "Enhancement":
        emode = st.sidebar.selectbox("Enhance", ["Histogram Equalization","Contrast Stretching","Sharpening"])
        img = orig_rgb.copy()
        if emode == "Histogram Equalization":
            processed = histogram_equalization(img)
        elif emode == "Contrast Stretching":
            lo = st.sidebar.slider("Low percentile", 0, 10, 2)
            hi = st.sidebar.slider("High percentile", 90, 100, 98)
            processed = contrast_stretch(img, lo, hi)
        elif emode == "Sharpening":
            amt = st.sidebar.slider("Amount", 0.0, 3.0, 1.0, 0.1)
            processed = sharpen(img, amt)

    elif mode == "Edge Detection":
        emode = st.sidebar.selectbox("Edge", ["Sobel","Canny","Laplacian"])
        gray = ensure_gray(orig_rgb)
        if emode == "Sobel":
            processed = sobel_edges(gray)
        elif emode == "Canny":
            t1 = st.sidebar.slider("Threshold1", 0, 255, 100)
            t2 = st.sidebar.slider("Threshold2", 0, 255, 200)
            processed = canny_edges(gray, t1, t2)
        else:
            processed = laplacian_edges(gray)

    elif mode == "Compression":
        fmt = st.sidebar.selectbox("Target format", ["png","jpg","bmp"])
        quality = st.sidebar.slider("JPEG Quality (if JPG)", 10, 100, 90)
        processed = orig_rgb.copy()
        # Show estimated file sizes
        buf_png = encode_format(processed, ".png")
        buf_jpg = cv2.imencode(".jpg", cv2.cvtColor(processed, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1].tobytes()
        buf_bmp = encode_format(processed, ".bmp")
        st.sidebar.write({"png_kb": round(len(buf_png)/1024,2),
                          "jpg_kb": round(len(buf_jpg)/1024,2),
                          "bmp_kb": round(len(buf_bmp)/1024,2)})

    elif mode == "Bitwise Ops":
        bmode = st.sidebar.selectbox("Bitwise", ["AND","OR","XOR","NOT"])
        img = orig_rgb.copy()
        # Create a mask shape for demo
        h, w = img.shape[:2]
        mask = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.circle(mask, (w//2, h//2), min(h,w)//4, (255,255,255), -1)
        if bmode == "AND":
            processed = bitwise_and(img, mask)
        elif bmode == "OR":
            processed = bitwise_or(img, mask)
        elif bmode == "XOR":
            processed = bitwise_xor(img, mask)
        else:
            processed = bitwise_not(img)

    elif mode == "Video (Bonus)":
        st.sidebar.info("Enable webcam to apply effects in real-time.")
        enable = st.sidebar.checkbox("Start Webcam")
        effect = st.sidebar.selectbox("Effect", ["Canny","Sobel","None"])
        if enable:
            frame = st.camera_input("Capture frame")
            if frame:
                file_bytes = np.asarray(bytearray(frame.getvalue()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                frame_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                if effect == "Canny":
                    gray = ensure_gray(frame_rgb)
                    processed = canny_edges(gray, 100, 200)
                elif effect == "Sobel":
                    gray = ensure_gray(frame_rgb)
                    processed = sobel_edges(gray)
                else:
                    processed = frame_rgb.copy()
        else:
            processed = orig_rgb.copy()

# Right panel display
with col2:
    st.subheader("Processed")
    if orig_rgb is not None and processed is not None:
        st.image(processed, use_container_width=True, clamp=True)
        compare = st.toggle("Split Screen Compare (Half/Half)", value=False)
        if compare:
            left = orig_rgb.copy()
            right = to_3channel(processed)
            if left.shape != right.shape:
                h = min(left.shape[0], right.shape[0])
                w = min(left.shape[1], right.shape[1])
                left = left[:h,:w]
                right = right[:h,:w]
            split = split_screen_compare(left, right)
            st.image(split, use_container_width=True, caption="Split Screen Comparison")
    elif orig_rgb is not None:
        st.info("Select an operation from the left panel.")

# Status bar (bottom)
st.markdown("---")
st.subheader("📊 Status")
if orig_rgb is not None:
    info = get_image_info(cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR), src_bytes, uploaded.name.split(".")[-1] if uploaded else None)
    st.write(f"Dimensions (H,W,C): {info['dimensions']} | File format: {info['file_format']} | File size: {info['file_size_kb']} KB | DPI/PPI: {info['dpi_ppi']}")
else:
    st.write("No image loaded.")

# Save processed
if orig_rgb is not None and 'processed' in locals() and processed is not None and save_btn:
    ext = f".{save_format}"
    buf = encode_format(to_3channel(processed), ext if ext != ".jpg" else ".jpg")
    if buf:
        st.download_button("Download processed image", data=buf, file_name=f"processed{ext}")
