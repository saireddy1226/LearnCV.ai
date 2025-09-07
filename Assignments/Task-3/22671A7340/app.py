
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2
import io
import os
from datetime import datetime
import tempfile
import math

st.set_page_config(page_title="Image Processing Toolkit", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# Utility functions
# ---------------------------
def to_bytes_io(img_pil, fmt="PNG", quality=95):
    buf = io.BytesIO()
    save_kwargs = {}
    if fmt.upper() == "JPEG":
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True
    img_pil.save(buf, format=fmt, **save_kwargs)
    buf.seek(0)
    return buf

def pil_to_cv2(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img_cv2):
    # cv2 image expected in BGR or single-channel
    if len(img_cv2.shape) == 2:
        return Image.fromarray(img_cv2)
    else:
        return Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

def format_bytes(size):
    # human readable size
    for unit in ['B','KB','MB','GB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"

def get_image_info(pil_img, uploaded_file=None):
    # Note: PIL doesn't always expose DPI; try to get from info
    info = {}
    info['format'] = pil_img.format or (uploaded_file.type.split("/")[-1].upper() if uploaded_file else "Unknown")
    info['mode'] = pil_img.mode
    info['size'] = pil_img.size  # (W,H)
    info['width'], info['height'] = pil_img.size
    info['channels'] = len(pil_img.getbands())
    info['dpi'] = pil_img.info.get('dpi', (72,72))
    # file size estimate: if uploaded_file present, get its size; else serialize PNG
    if uploaded_file:
        try:
            uploaded_file.seek(0, os.SEEK_END)
            size = uploaded_file.tell()
            uploaded_file.seek(0)
            info['filesize'] = size
        except Exception:
            buf = to_bytes_io(pil_img, fmt="PNG")
            info['filesize'] = len(buf.getvalue())
    else:
        buf = to_bytes_io(pil_img, fmt="PNG")
        info['filesize'] = len(buf.getvalue())
    return info

# ---------------------------
# Color conversion helpers (manual + cv2)
# ---------------------------
def rgb_to_grayscale_manual(img_rgb):
    # img_rgb: numpy RGB float or uint8
    r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
    gray = 0.2989*r + 0.5870*g + 0.1140*b
    return gray.astype(np.uint8)

def rgb_to_ycbcr_manual(img_rgb):
    # input RGB uint8 -> output YCbCr uint8
    M = np.array([[ 0.299,  0.587,  0.114],
                  [-0.168736, -0.331264, 0.5],
                  [0.5, -0.418688, -0.081312]])
    arr = img_rgb.astype(np.float32)
    ycbcr = arr.dot(M.T)
    ycbcr[:,:,1:] += 128.0
    ycbcr = np.clip(ycbcr,0,255)
    return ycbcr.astype(np.uint8)

# ---------------------------
# Geometry transforms
# ---------------------------
def rotate_image_cv(img, angle, center=None, scale=1.0):
    (h, w) = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated

def scale_image_cv(img, fx, fy):
    return cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)

def translate_image_cv(img, tx, ty):
    M = np.float32([[1,0,tx],[0,1,ty]])
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def affine_transform_cv(img, src_pts, dst_pts):
    M = cv2.getAffineTransform(np.float32(src_pts), np.float32(dst_pts))
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def perspective_transform_cv(img, src_pts, dst_pts, out_size=None):
    M = cv2.getPerspectiveTransform(np.float32(src_pts), np.float32(dst_pts))
    h, w = img.shape[:2]
    if out_size is None:
        out_size = (w, h)
    return cv2.warpPerspective(img, M, out_size, borderMode=cv2.BORDER_REFLECT)

# ---------------------------
# Filters and morphology
# ---------------------------
def apply_filter(img, filter_name, ksize=3):
    if filter_name == "Gaussian":
        return cv2.GaussianBlur(img, (ksize,ksize), 0)
    elif filter_name == "Median":
        return cv2.medianBlur(img, ksize)
    elif filter_name == "Mean":
        return cv2.blur(img, (ksize,ksize))
    else:
        return img

def apply_morph(img, op_name, ksize=3, iterations=1):
    kernel = np.ones((ksize,ksize), np.uint8)
    if op_name == "Dilation":
        return cv2.dilate(img, kernel, iterations=iterations)
    if op_name == "Erosion":
        return cv2.erode(img, kernel, iterations=iterations)
    if op_name == "Opening":
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    if op_name == "Closing":
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img

# ---------------------------
# Enhancement & edges
# ---------------------------
def histogram_equalization_color(img):
    # img in BGR
    if len(img.shape) == 2:
        return cv2.equalizeHist(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def contrast_stretching(img):
    # linear contrast stretching per channel
    out = np.zeros_like(img)
    for c in range(img.shape[2]) if img.ndim==3 else range(1):
        channel = img[:,:,c] if img.ndim==3 else img
        minv = channel.min()
        maxv = channel.max()
        if maxv - minv == 0:
            stretched = channel
        else:
            stretched = ((channel - minv) * 255.0 / (maxv - minv)).astype(np.uint8)
        if img.ndim==3:
            out[:,:,c] = stretched
        else:
            out = stretched
    return out

def sharpen_image(img):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img, -1, kernel)

def edge_detection(img, method="Canny", **kwargs):
    gray = img if len(img.shape)==2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if method == "Canny":
        low = kwargs.get("low",100)
        high = kwargs.get("high",200)
        return cv2.Canny(gray, low, high)
    elif method == "Sobel":
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kwargs.get("ksize",3))
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kwargs.get("ksize",3))
        sob = np.sqrt(sx*sx + sy*sy)
        sob = np.clip(sob, 0, 255).astype(np.uint8)
        return sob
    elif method == "Laplacian":
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
        return lap
    return gray

# ---------------------------
# Bitwise ops
# ---------------------------
def bitwise_ops(img1, img2, op="AND"):
    # expects BGR uint8 same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    if op == "AND":
        return cv2.bitwise_and(img1, img2)
    if op == "OR":
        return cv2.bitwise_or(img1, img2)
    if op == "XOR":
        return cv2.bitwise_xor(img1, img2)
    if op == "NOT":
        return cv2.bitwise_not(img1)
    return img1

# ---------------------------
# App UI and logic
# ---------------------------
st.title("🖼 Image Processing Toolkit — Assignment 3")
st.markdown("Streamlit GUI to apply basic image processing operations using OpenCV. Deadline: *Sep 8, 2025*")

# Menu / File actions
menu_col1, menu_col2, menu_col3 = st.columns([1,1,6])
with menu_col1:
    if st.button("Open 🔍"):
        st.experimental_rerun()  # use streamlit upload widget below to select file
with menu_col2:
    pass
# Sidebar: upload + operations
with st.sidebar:
    st.header("File")
    uploaded_file = st.file_uploader("Upload an image (PNG/JPG/BMP)", type=['png','jpg','jpeg','bmp'])
    if uploaded_file:
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Could not open image: {e}")
            pil_image = None
    else:
        pil_image = None

    st.markdown("---")
    st.header("Operations")
    op_mode = st.selectbox("Choose category", [
        "Image Info", "Color Conversion", "Transformations", "Filters & Morphology",
        "Enhancement & Edge Detection", "Bitwise Ops", "Compression & Save", "Webcam / Real-time"
    ])

    # Shared sliders (will be used in sections as needed)
    st.markdown("### Common sliders (appear when needed)")
    ksize = st.slider("Kernel size (odd)", min_value=1, max_value=31, value=3, step=2)
    angle = st.slider("Rotation angle (deg)", -180, 180, 0)
    scale_factor = st.slider("Scaling factor", 0.1, 3.0, 1.0, step=0.05)
    tx = st.slider("Translate X (px)", -500, 500, 0)
    ty = st.slider("Translate Y (px)", -500, 500, 0)

    st.markdown("---")
    st.header("Display Options")
    show_side_by_side = st.checkbox("Show side-by-side (two columns)", value=True)
    compare_mode = st.checkbox("Split-screen comparison (half original, half processed)", value=False)
    auto_apply = st.checkbox("Auto apply on change", value=True)
    st.markdown("---")
    st.header("Save / Export")
    out_fmt = st.selectbox("Save format", ["PNG","JPEG","BMP"])
    jpeg_quality = st.slider("JPEG Quality", 30, 100, 95) if out_fmt=="JPEG" else None

# Prepare images
if pil_image is not None:
    original_pil = pil_image.copy()
    original_cv = pil_to_cv2(original_pil)
    working_cv = original_cv.copy()
    last_op_name = "Original"
else:
    original_pil = None
    original_cv = None
    working_cv = None
    last_op_name = "None"

# Right panel display area
left_col, right_col = st.columns([1,1]) if show_side_by_side else (st.container(), st.container())

def display_images(orig_pil, proc_pil, caption_left="Original", caption_right="Processed"):
    # If split compare mode: create half-half image
    if compare_mode and orig_pil and proc_pil:
        # ensure same size
        proc_resized = proc_pil.resize(orig_pil.size)
        w,h = orig_pil.size
        half = w//2
        combined = Image.new("RGB", (w, h))
        combined.paste(orig_pil.crop((0,0,half,h)), (0,0))
        combined.paste(proc_resized.crop((half,0,w,h)), (half,0))
        st.image(combined, caption="Split: left=Original | right=Processed", use_column_width='always')
        return

    # Normal side-by-side or single
    if show_side_by_side:
        c1, c2 = st.columns(2)
        with c1:
            if orig_pil:
                st.image(orig_pil, caption=caption_left, use_column_width='always')
            else:
                st.info("Upload an image to begin.")
        with c2:
            if proc_pil:
                st.image(proc_pil, caption=caption_right, use_column_width='always')
            else:
                st.info("No processed image yet.")
    else:
        if orig_pil:
            st.image(orig_pil, caption=caption_left, use_column_width='always')
        if proc_pil:
            st.image(proc_pil, caption=caption_right, use_column_width='always')

# Operation-specific UI and actions
processed_cv = None
manual_info_outputs = None

if op_mode == "Image Info":
    st.header("Image Information")
    if original_pil is None:
        st.info("Upload an image to show metadata.")
    else:
        info = get_image_info(original_pil, uploaded_file)
        st.write("*Format:*", info['format'])
        st.write("*Mode / Channels:*", info['mode'], f"({info['channels']} channels)")
        st.write("*Dimensions (W x H):*", f"{info['width']} x {info['height']}")
        st.write("*DPI:*", info['dpi'])
        st.write("*File size:*", format_bytes(info['filesize']))
        # show pixel stats
        arr = np.array(original_pil)
        st.write("*Min / Max / Mean (per channel)*")
        if arr.ndim == 3:
            mins = arr.reshape(-1,3).min(axis=0)
            maxs = arr.reshape(-1,3).max(axis=0)
            means = arr.reshape(-1,3).mean(axis=0)
            st.write(f"R: min {mins[0]}, max {maxs[0]}, mean {means[0]:.2f}")
            st.write(f"G: min {mins[1]}, max {maxs[1]}, mean {means[1]:.2f}")
            st.write(f"B: min {mins[2]}, max {maxs[2]}, mean {means[2]:.2f}")
        else:
            st.write(f"Min {arr.min()}, Max {arr.max()}, Mean {arr.mean():.2f}")

    processed_cv = original_cv.copy() if original_cv is not None else None

elif op_mode == "Color Conversion":
    st.header("Color Conversion")
    conv = st.selectbox("Conversion", [
        "RGB ↔ BGR (cv2)", "RGB → Grayscale (manual)", "RGB → Grayscale (cv2)",
        "RGB → HSV (cv2)", "RGB → YCbCr (manual)", "RGB → YCbCr (cv2)"
    ])
    if original_cv is None:
        st.info("Upload an image to apply conversions.")
    else:
        if conv == "RGB ↔ BGR (cv2)":
            # display swap channels
            processed_cv = cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB)
            last_op_name = "BGR→RGB"
        elif conv == "RGB → Grayscale (manual)":
            rgb = cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB)
            gray = rgb_to_grayscale_manual(rgb)
            processed_cv = gray
            last_op_name = "Grayscale (manual)"
        elif conv == "RGB → Grayscale (cv2)":
            processed_cv = cv2.cvtColor(original_cv, cv2.COLOR_BGR2GRAY)
            last_op_name = "Grayscale (cv2)"
        elif conv == "RGB → HSV (cv2)":
            hsv = cv2.cvtColor(original_cv, cv2.COLOR_BGR2HSV)
            # display HSV as an image by converting back to BGR for visualization
            processed_cv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            last_op_name = "HSV (visualized)"
        elif conv == "RGB → YCbCr (manual)":
            rgb = cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB)
            ycbcr = rgb_to_ycbcr_manual(rgb)
            processed_cv = cv2.cvtColor(ycbcr, cv2.COLOR_RGB2BGR)
            last_op_name = "YCbCr (manual)"
        elif conv == "RGB → YCbCr (cv2)":
            ycrcb = cv2.cvtColor(original_cv, cv2.COLOR_BGR2YCrCb)
            # visualize by converting back
            processed_cv = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            last_op_name = "YCbCr (cv2)"

elif op_mode == "Transformations":
    st.header("Transformations")
    t = st.selectbox("Choose transform", ["Rotate", "Scale", "Translate", "Affine Transform", "Perspective Transform"])
    if original_cv is None:
        st.info("Upload an image to transform.")
    else:
        if t == "Rotate":
            processed_cv = rotate_image_cv(original_cv, angle=angle, scale=scale_factor)
            last_op_name = f"Rotate {angle}°"
        elif t == "Scale":
            processed_cv = scale_image_cv(original_cv, fx=scale_factor, fy=scale_factor)
            last_op_name = f"Scale x{scale_factor:.2f}"
        elif t == "Translate":
            processed_cv = translate_image_cv(original_cv, tx, ty)
            last_op_name = f"Translate ({tx}px, {ty}px)"
        elif t == "Affine Transform":
            st.write("Drag sample source/destination points (example).")
            h,w = original_cv.shape[:2]
            src_pts = st.text_input("Source triangle (x1,y1;x2,y2;x3,y3)", value=f"0,0;{w-1},0;0,{h-1}")
            dst_pts = st.text_input("Destination triangle (x1,y1;x2,y2;x3,y3)", value=f"0,0;{int(0.8*(w-1))},int(0.2*{h-1});int(0.2*{w-1}),{h-1}")
            try:
                sp = [tuple(map(float,s.split(","))) for s in src_pts.split(";")]
                dp = [tuple(map(float,s.split(","))) for s in dst_pts.split(";")]
                processed_cv = affine_transform_cv(original_cv, sp, dp)
                last_op_name = "Affine"
            except Exception as e:
                st.error("Invalid points format. Use x,y;x,y;x,y")
                processed_cv = original_cv.copy()
        elif t == "Perspective Transform":
            st.write("Use 4 source and destination points: x,y;x,y;x,y;x,y")
            h,w = original_cv.shape[:2]
            src_pts = st.text_input("Source quad", value=f"0,0;{w-1},0;{w-1},{h-1};0,{h-1}")
            dst_pts = st.text_input("Dest quad (example moved)", value=f"0,0;{w-1},0;{int(0.9*(h-1))},{h-1};0,{h-1}")
            try:
                sp = [tuple(map(float,s.split(","))) for s in src_pts.split(";")]
                dp = [tuple(map(float,s.split(","))) for s in dst_pts.split(";")]
                processed_cv = perspective_transform_cv(original_cv, sp, dp)
                last_op_name = "Perspective"
            except Exception as e:
                st.error("Invalid points format. Use x,y;...")
                processed_cv = original_cv.copy()

elif op_mode == "Filters & Morphology":
    st.header("Filters & Morphology")
    fop = st.selectbox("Filter / Morph operation", ["Smoothing (Gaussian/Median/Mean)", "Edge filters (Sobel/Laplacian)", "Morphology (Dilation/Erosion/Opening/Closing)"])
    if original_cv is None:
        st.info("Upload an image to apply.")
    else:
        if fop.startswith("Smoothing"):
            ftype = st.selectbox("Type", ["Gaussian","Median","Mean"])
            processed_cv = apply_filter(original_cv, ftype, ksize=ksize)
            last_op_name = f"{ftype} k={ksize}"
        elif fop.startswith("Edge filters"):
            et = st.selectbox("Edge filter", ["Sobel","Laplacian"])
            if et == "Sobel":
                sx = cv2.Sobel(cv2.cvtColor(original_cv, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize=ksize)
                sy = cv2.Sobel(cv2.cvtColor(original_cv, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 0, 1, ksize=ksize)
                sob = np.sqrt(sx*sx + sy*sy)
                sob = np.clip(sob, 0, 255).astype(np.uint8)
                processed_cv = sob
                last_op_name = "Sobel"
            else:
                lap = cv2.Laplacian(cv2.cvtColor(original_cv, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
                lap = np.clip(np.abs(lap),0,255).astype(np.uint8)
                processed_cv = lap
                last_op_name = "Laplacian"
        else:
            mop = st.selectbox("Morph op", ["Dilation","Erosion","Opening","Closing"])
            iter_count = st.slider("Iterations", 1, 10, 1)
            processed_cv = apply_morph(original_cv, mop, ksize=ksize, iterations=iter_count)
            last_op_name = f"{mop} k={ksize} it={iter_count}"

elif op_mode == "Enhancement & Edge Detection":
    st.header("Enhancement & Edge Detection")
    eop = st.selectbox("Operation", ["Histogram Equalization", "Contrast Stretching", "Sharpening", "Canny Edge Detection", "Sobel Edge", "Laplacian Edge"])
    if original_cv is None:
        st.info("Upload image first.")
    else:
        if eop == "Histogram Equalization":
            processed_cv = histogram_equalization_color(original_cv)
            last_op_name = "Hist Equalization"
        elif eop == "Contrast Stretching":
            processed_cv = contrast_stretching(original_cv)
            last_op_name = "Contrast Stretching"
        elif eop == "Sharpening":
            processed_cv = sharpen_image(original_cv)
            last_op_name = "Sharpen"
        elif eop == "Canny Edge Detection":
            low = st.slider("Canny low threshold", 0, 500, 100)
            high = st.slider("Canny high threshold", 0, 500, 200)
            processed_cv = edge_detection(original_cv, method="Canny", low=low, high=high)
            last_op_name = f"Canny {low},{high}"
        elif eop == "Sobel Edge":
            processed_cv = edge_detection(original_cv, method="Sobel", ksize=ksize)
            last_op_name = "Sobel"
        elif eop == "Laplacian Edge":
            processed_cv = edge_detection(original_cv, method="Laplacian")
            last_op_name = "Laplacian"

elif op_mode == "Bitwise Ops":
    st.header("Bitwise Operations")
    bop = st.selectbox("Bitwise op", ["AND","OR","XOR","NOT"])
    st.write("Second image to combine (optional). If not provided, operation uses original image and its inverted/itself as appropriate.")
    uploaded_file2 = st.file_uploader("Upload second image (for binary ops)", type=['png','jpg','jpeg','bmp'], key="second_image")
    second_cv = None
    if uploaded_file2:
        try:
            second_pil = Image.open(uploaded_file2).convert("RGB")
            second_cv = pil_to_cv2(second_pil)
        except Exception as e:
            st.warning("Could not open second image.")
    if original_cv is None:
        st.info("Upload base image first.")
    else:
        if bop == "NOT":
            processed_cv = bitwise_ops(original_cv, None, op="NOT")
            last_op_name = "NOT"
        else:
            if second_cv is None:
                # create a shifted version of original as second
                second_cv = translate_image_cv(original_cv, tx=50, ty=50)
            processed_cv = bitwise_ops(original_cv, second_cv, op=bop)
            last_op_name = f"Bitwise {bop}"

elif op_mode == "Compression & Save":
    st.header("Compression & File Handling")
    if original_pil is None:
        st.info("Upload image to test saving & compression.")
    else:
        st.write("Select format and check resulting filesize. Use Save button to download processed image.")
        # show size choices
        formats = ["PNG","JPEG","BMP"]
        fmt = st.selectbox("Format to test", formats)
        qual = st.slider("JPEG Quality (when JPEG)", 30, 100, 95) if fmt=="JPEG" else None
        # process: default is current original
        processed_cv = original_cv.copy()
        # save to buffer and show size
        tmp_pil = cv2_to_pil(processed_cv)
        buf = to_bytes_io(tmp_pil, fmt=fmt, quality=qual if qual else 95)
        st.write("Saved size:", format_bytes(len(buf.getvalue())))
        st.download_button("Save processed image", data=buf, file_name=f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{fmt.lower()}", mime=f"image/{fmt.lower()}")

elif op_mode == "Webcam / Real-time":
    st.header("Real-time Webcam Mode")
    st.warning("Webcam in Streamlit may not work in all environments (browser permission required). This will use OpenCV to capture if available.")
    run_cam = st.checkbox("Start webcam", False)
    selected_op = st.selectbox("Realtime op", ["None","Grayscale","Canny","Gaussian Blur","Sharpen"])
    if run_cam:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while run_cam:
            ret, frame = cap.read()
            if not ret:
                st.write("No camera frames. Stop and check camera.")
                break
            if selected_op == "Grayscale":
                out = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
            elif selected_op == "Canny":
                out_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(out_gray, 100, 200)
                out = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            elif selected_op == "Gaussian Blur":
                out = cv2.GaussianBlur(frame, (ksize|1, ksize|1), 0)
            elif selected_op == "Sharpen":
                out = sharpen_image(frame)
            else:
                out = frame
            stframe.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), channels="RGB")
            # break loop if checkbox turned off
            run_cam = st.checkbox("Start webcam", value=True)
        cap.release()
        cv2.destroyAllWindows()

# Convert processed_cv to PIL for display/save
processed_pil = None
if processed_cv is not None:
    if len(processed_cv.shape) == 2:
        # grayscale single channel
        processed_pil = Image.fromarray(processed_cv)
    else:
        processed_pil = cv2_to_pil(processed_cv)

# Display images
display_images(original_pil, processed_pil, caption_left="Original", caption_right=f"Processed: {last_op_name}")

# Status bar
st.markdown("---")
st.write("### Status")
if original_pil:
    info = get_image_info(original_pil, uploaded_file)
    st.write(f"Dimensions: {info['width']} x {info['height']} (W x H)")
    st.write(f"Channels / Mode: {info['channels']} / {info['mode']}")
    st.write(f"Format: {info['format']} • File size: {format_bytes(info['filesize'])} • DPI: {info['dpi']}")
else:
    st.write("No image loaded.")

if processed_pil:
    # allow saving processed image
    buf_out = to_bytes_io(processed_pil, fmt=out_fmt, quality=jpeg_quality if jpeg_quality else 95)
    st.download_button("Save Processed Image", data=buf_out, file_name=f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{out_fmt.lower()}", mime=f"image/{out_fmt.lower()}")

# Extra: provide ability to compare original and processed file sizes in chosen formats
st.markdown("---")
st.subheader("File size comparison (Original vs Processed)")
if original_pil:
    fmt = st.selectbox("Compare save format", ["PNG","JPEG","BMP"], key="compare_fmt")
    q = st.slider("JPEG Quality for comparison", 30, 100, 95, key="compare_quality") if fmt=="JPEG" else None
    buf_orig = to_bytes_io(original_pil, fmt=fmt, quality=q if q else 95)
    if processed_pil:
        buf_proc = to_bytes_io(processed_pil, fmt=fmt, quality=q if q else 95)
        st.write("Original size:", format_bytes(len(buf_orig.getvalue())))
        st.write("Processed size:", format_bytes(len(buf_proc.getvalue())))
        ratio = (len(buf_proc.getvalue()) / max(1, len(buf_orig.getvalue()))) * 100
        st.write(f"Processed is {ratio:.1f}% of original size.")
    else:
        st.write("No processed image to compare.")

# Footer / help
st.markdown("---")
st.info("Tips: Use sliders and options in sidebar. Use 'Split-screen comparison' to see half original half processed. Save processed image with the download button.")
