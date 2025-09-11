"""
Image Processing & Analysis Toolkit
Streamlit + OpenCV + NumPy

Single-file Streamlit app implementing:
- Image upload and info
- Color conversions (RGB/BGR/HSV/YCbCr/Gray)
- Geometric transforms: rotate, scale, translate, affine, perspective
- Bitwise ops (requires two images)
- Filtering & morphology
- Enhancement & edge detection
- Compression (save in JPG/PNG/BMP) and size comparison
- Live webcam snapshot processing
- Sliders for kernel size, angle, thresholds
- Split comparison mode and download/save capabilities

Run:
    pip install streamlit opencv-python-headless numpy pillow matplotlib
    streamlit run ImageProcessingToolkit_app.py

Note: Use opencv-python-headless on servers; opencv-python on local machines with GUI.
"""

import io
import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import math

st.set_page_config(page_title="Image Processing Toolkit", layout="wide")

# ----------------------- Utilities -----------------------

def to_cv2(img_pil: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR format (uint8)."""
    img = np.array(img_pil)
    if img.ndim == 2:
        return img
    # PIL gives RGB
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def to_pil(img_cv: np.ndarray) -> Image.Image:
    """Convert OpenCV image (BGR or grayscale) to PIL RGB image."""
    if img_cv is None:
        return None
    if img_cv.ndim == 2:
        return Image.fromarray(img_cv)
    # assume BGR
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


def get_image_info(img_cv: np.ndarray, fname: str = None) -> dict:
    info = {}
    if img_cv is None:
        return info
    h, w = img_cv.shape[:2]
    c = 1 if img_cv.ndim == 2 else img_cv.shape[2]
    info['width'] = w
    info['height'] = h
    info['channels'] = c
    info['shape'] = img_cv.shape
    if fname and os.path.exists(fname):
        info['file_size'] = os.path.getsize(fname)
        info['format'] = os.path.splitext(fname)[1].lower().replace('.', '')
    return info


def pil_bytes(img_pil: Image.Image, fmt: str = 'PNG', quality: int = 95) -> bytes:
    buf = io.BytesIO()
    save_kwargs = {'format': fmt}
    if fmt.upper() in ('JPEG', 'JPG'):
        save_kwargs['quality'] = quality
    img_pil.save(buf, **save_kwargs)
    return buf.getvalue()


# Color conversion helpers (mathematical fallback if needed)

def rgb2ycbcr_numpy(img_rgb: np.ndarray) -> np.ndarray:
    # img_rgb is RGB uint8
    arr = img_rgb.astype(np.float32)
    R = arr[..., 0]
    G = arr[..., 1]
    B = arr[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B
    out = np.stack([Y, Cb, Cr], axis=-1)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


# ----------------------- Streamlit UI -----------------------

st.title("📸 Image Processing & Analysis Toolkit")
st.markdown("A beginner→advanced toolkit using Streamlit + OpenCV. Upload an image, apply operations and compare results.")

# Top menu (simple emulation using expander)
with st.expander("File"):
    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        uploaded = st.file_uploader("Open → Upload an image", type=['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'])
    with colf2:
        save_format = st.selectbox("Save format", ['PNG', 'JPEG', 'BMP'])
        save_quality = st.slider("JPEG Quality", 50, 100, 95)
        if st.button("Save Processed Image"):
            # handled later when processed exists
            st.session_state['_save_requested'] = True
    with colf3:
        if st.button("Exit"):
            st.write("To exit, close the Streamlit tab/window.")

# Initialize session state
if 'orig' not in st.session_state:
    st.session_state['orig'] = None
if 'proc' not in st.session_state:
    st.session_state['proc'] = None
if '_save_requested' not in st.session_state:
    st.session_state['_save_requested'] = False
if 'second_img' not in st.session_state:
    st.session_state['second_img'] = None

# Left panel - operations
with st.sidebar:
    st.header("Operations")
    op_group = st.selectbox("Category", [
        'Image Info', 'Color Conversions', 'Transformations', 'Bitwise Ops',
        'Filtering & Morphology', 'Enhancement', 'Edge Detection', 'Compression', 'Extras'
    ])

    # Common params
    split_mode = st.checkbox('Comparison mode: split original/processed', value=False)
    show_histograms = st.checkbox('Show histograms', value=False)

    # Parameters common sliders (appear when needed)
    kernel_size = st.slider('Kernel size (odd)', 1, 31, 3, step=2)
    angle = st.slider('Rotation angle (degrees)', -180, 180, 0)
    scale = st.slider('Scaling factor', 0.1, 3.0, 1.0)
    tx = st.slider('Translate X (pixels)', -500, 500, 0)
    ty = st.slider('Translate Y (pixels)', -500, 500, 0)

    # operation selection within group
    if op_group == 'Image Info':
        info_opts = st.multiselect('Show:', ['Resolution', 'Shape', 'DPI (if available)', 'File format', 'Color channels'])
    elif op_group == 'Color Conversions':
        color_op = st.selectbox('Convert:', ['RGB → BGR', 'BGR → RGB', 'RGB → HSV', 'RGB → YCbCr (math)', 'RGB → Grayscale', 'HSV → RGB'])
    elif op_group == 'Transformations':
        trans_op = st.selectbox('Transform:', ['Rotate', 'Scale', 'Translate', 'Affine Transform', 'Perspective Transform'])
        if trans_op == 'Affine Transform':
            st.info('Affine: drag three source->destination points on image preview is not available; we provide sample transforms')
            affine_scale = st.slider('Affine: shear/scale factor', 0.5, 1.5, 1.0)
        if trans_op == 'Perspective Transform':
            persp_amount = st.slider('Perspective warp amount', 0.0, 1.0, 0.2)
    elif op_group == 'Bitwise Ops':
        bit_op = st.selectbox('Bitwise:', ['AND', 'OR', 'XOR', 'NOT'])
        st.write('Second image (for AND/OR/XOR)')
        second_uploaded = st.file_uploader('Upload second image (same size recommended)', type=['png', 'jpg', 'jpeg', 'bmp'], key='second')
    elif op_group == 'Filtering & Morphology':
        filter_op = st.selectbox('Filter/Morph:', ['Gaussian Blur', 'Mean Blur', 'Median Blur', 'Sobel', 'Laplacian', 'Dilation', 'Erosion', 'Opening', 'Closing'])
        iter_morph = st.slider('Morph iterations', 1, 10, 1)
    elif op_group == 'Enhancement':
        enhance_op = st.selectbox('Enhancement:', ['Histogram Equalization (grayscale)', 'CLAHE (color-aware)', 'Contrast Stretching', 'Sharpening'])
    elif op_group == 'Edge Detection':
        edge_op = st.selectbox('Edge:', ['Canny', 'Sobel (combined)', 'Laplacian'])
        low_thr = st.slider('Canny low threshold', 0, 255, 50)
        high_thr = st.slider('Canny high threshold', 0, 255, 150)
    elif op_group == 'Compression':
        st.write('Choose save format and compare sizes')
        comp_format = st.selectbox('Compare save as', ['PNG', 'JPEG', 'BMP'])
    elif op_group == 'Extras':
        extras = st.multiselect('Extras', ['Split Half Comparison', 'Webcam snapshot mode', 'Download processed'])

    apply_button = st.button('Apply Operation')
    reset_button = st.button('Reset processed image')

# Load images
if uploaded is not None:
    try:
        pil = Image.open(uploaded).convert('RGB')
        img_cv = to_cv2(pil)
        st.session_state['orig'] = img_cv
        st.session_state['proc'] = img_cv.copy()
        st.session_state['orig_pil'] = pil
        st.success('Image loaded: {} x {}'.format(img_cv.shape[1], img_cv.shape[0]))
    except Exception as e:
        st.error('Error loading image: ' + str(e))

if 'second' in st.session_state and st.session_state['second'] is not None:
    try:
        second = Image.open(st.session_state['second']).convert('RGB')
        st.session_state['second_img'] = to_cv2(second)
    except Exception:
        st.session_state['second_img'] = None

if reset_button:
    st.session_state['proc'] = st.session_state.get('orig', None)

# Apply selected operation
if apply_button:
    src = st.session_state.get('orig', None)
    if src is None:
        st.warning('Please upload an image first.')
    else:
        proc = src.copy()
        try:
            if op_group == 'Image Info':
                st.write('Image information:')
                info = get_image_info(proc)
                st.json(info)
            elif op_group == 'Color Conversions':
                if color_op == 'RGB → BGR':
                    # our src is BGR by default, convert to RGB then back to BGR demonstration
                    rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
                    proc = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                elif color_op == 'BGR → RGB':
                    proc = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
                elif color_op == 'RGB → HSV':
                    proc = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
                elif color_op == 'RGB → YCbCr (math)':
                    rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
                    ycbcr = rgb2ycbcr_numpy(rgb)
                    proc = cv2.cvtColor(ycbcr, cv2.COLOR_RGB2BGR)
                elif color_op == 'RGB → Grayscale':
                    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
                    proc = gray
                elif color_op == 'HSV → RGB':
                    hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
                    proc = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            elif op_group == 'Transformations':
                h, w = proc.shape[:2]
                if trans_op == 'Rotate':
                    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
                    proc = cv2.warpAffine(proc, M, (w, h))
                elif trans_op == 'Scale':
                    nw = int(w * scale)
                    nh = int(h * scale)
                    proc = cv2.resize(proc, (nw, nh), interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC)
                elif trans_op == 'Translate':
                    M = np.float32([[1, 0, tx], [0, 1, ty]])
                    proc = cv2.warpAffine(proc, M, (w, h))
                elif trans_op == 'Affine Transform':
                    # sample affine: shift corners
                    pts1 = np.float32([[0,0],[w-1,0],[0,h-1]])
                    pts2 = np.float32([[0,0],[int((w-1)*affine_scale),0],[0,int((h-1)*affine_scale)]])
                    M = cv2.getAffineTransform(pts1, pts2)
                    proc = cv2.warpAffine(proc, M, (w, h))
                elif trans_op == 'Perspective Transform':
                    margin = int(min(w,h) * persp_amount)
                    pts1 = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
                    pts2 = np.float32([[margin,margin],[w-1-margin,margin],[w-1-margin,h-1-margin],[margin,h-1-margin]])
                    M = cv2.getPerspectiveTransform(pts1, pts2)
                    proc = cv2.warpPerspective(proc, M, (w, h))

            elif op_group == 'Bitwise Ops':
                sec = st.session_state.get('second_img', None)
                if bit_op == 'NOT':
                    proc = cv2.bitwise_not(proc)
                else:
                    if sec is None:
                        st.warning('Provide second image for this operation. Using original again as fallback.')
                        sec = proc
                    # resize second image to match
                    if sec.shape[:2] != proc.shape[:2]:
                        sec = cv2.resize(sec, (proc.shape[1], proc.shape[0]))
                    if bit_op == 'AND':
                        proc = cv2.bitwise_and(proc, sec)
                    elif bit_op == 'OR':
                        proc = cv2.bitwise_or(proc, sec)
                    elif bit_op == 'XOR':
                        proc = cv2.bitwise_xor(proc, sec)

            elif op_group == 'Filtering & Morphology':
                if filter_op == 'Gaussian Blur':
                    proc = cv2.GaussianBlur(proc, (kernel_size, kernel_size), 0)
                elif filter_op == 'Mean Blur':
                    proc = cv2.blur(proc, (kernel_size, kernel_size))
                elif filter_op == 'Median Blur':
                    proc = cv2.medianBlur(proc, kernel_size)
                elif filter_op == 'Sobel':
                    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
                    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
                    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
                    sob = cv2.magnitude(sx, sy)
                    sob = np.clip((sob/np.max(sob))*255, 0, 255).astype(np.uint8)
                    proc = sob
                elif filter_op == 'Laplacian':
                    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
                    lap = cv2.Laplacian(gray, cv2.CV_64F)
                    lap = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
                    proc = lap
                else:
                    kernel = np.ones((kernel_size,kernel_size), np.uint8)
                    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
                    if filter_op == 'Dilation':
                        proc = cv2.dilate(gray, kernel, iterations=iter_morph)
                    elif filter_op == 'Erosion':
                        proc = cv2.erode(gray, kernel, iterations=iter_morph)
                    elif filter_op == 'Opening':
                        proc = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=iter_morph)
                    elif filter_op == 'Closing':
                        proc = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=iter_morph)

            elif op_group == 'Enhancement':
                if enhance_op == 'Histogram Equalization (grayscale)':
                    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
                    proc = cv2.equalizeHist(gray)
                elif enhance_op == 'CLAHE (color-aware)':
                    lab = cv2.cvtColor(proc, cv2.COLOR_BGR2LAB)
                    l,a,b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    cl = clahe.apply(l)
                    merged = cv2.merge((cl,a,b))
                    proc = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
                elif enhance_op == 'Contrast Stretching':
                    in_min = np.percentile(proc, 2)
                    in_max = np.percentile(proc, 98)
                    proc = np.clip((proc - in_min) * 255.0 / (in_max - in_min), 0, 255).astype(np.uint8)
                elif enhance_op == 'Sharpening':
                    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
                    proc = cv2.filter2D(proc, -1, kernel)

            elif op_group == 'Edge Detection':
                if edge_op == 'Canny':
                    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
                    proc = cv2.Canny(gray, low_thr, high_thr)
                elif edge_op == 'Sobel (combined)':
                    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
                    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
                    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
                    mag = cv2.magnitude(sx, sy)
                    proc = np.clip((mag/np.max(mag))*255, 0, 255).astype(np.uint8)
                elif edge_op == 'Laplacian':
                    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
                    lap = cv2.Laplacian(gray, cv2.CV_64F)
                    proc = np.clip(np.abs(lap), 0, 255).astype(np.uint8)

            elif op_group == 'Compression':
                # We'll prepare a compressed bytes to compare sizes
                pil_orig = to_pil(proc)
                bts = pil_bytes(pil_orig, fmt=comp_format, quality=save_quality)
                st.write(f"Estimated size when saved as {comp_format}: {len(bts)} bytes")

            elif op_group == 'Extras':
                if 'Webcam snapshot mode' in extras:
                    cam_img = st.camera_input('Take a webcam snapshot (processed)')
                    if cam_img is not None:
                        cam_pil = Image.open(cam_img).convert('RGB')
                        cam_cv = to_cv2(cam_pil)
                        # apply simple selected default op: e.g., grayscale
                        proc = cv2.cvtColor(cam_cv, cv2.COLOR_BGR2GRAY)
                        st.session_state['proc'] = proc
                        st.success('Processed webcam snapshot')
                if 'Split Half Comparison' in extras:
                    split_mode = True
                if 'Download processed' in extras:
                    st.session_state['_download_enabled'] = True

            # finalize processed
            st.session_state['proc'] = proc
            st.success('Operation applied')
        except Exception as e:
            st.error('Error applying operation: ' + str(e))

# Display area (two columns)
col1, col2 = st.columns(2)
with col1:
    st.subheader('Original Image')
    orig = st.session_state.get('orig', None)
    if orig is not None:
        orig_pil = to_pil(orig)
        st.image(orig_pil, use_column_width=True)
    else:
        st.info('Upload an image to begin')

with col2:
    st.subheader('Processed Image')
    proc = st.session_state.get('proc', None)
    if proc is not None:
        # If grayscale single channel -> show directly
        if proc.ndim == 2:
            st.image(proc, clamp=True, channels='GRAY', use_column_width=True)
        else:
            # assume BGR
            st.image(to_pil(proc), use_column_width=True)
    else:
        st.info('Processed image will appear here')

# If split mode requested, show split comparison
if split_mode and st.session_state.get('orig', None) is not None and st.session_state.get('proc', None) is not None:
    try:
        o = st.session_state['orig']
        p = st.session_state['proc']
        # ensure same size & 3 channels for split
        if p.ndim == 2:
            p_rgb = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
        else:
            p_rgb = p
        o_rgb = o if o.ndim==3 else cv2.cvtColor(o, cv2.COLOR_GRAY2BGR)
        p_resized = cv2.resize(p_rgb, (o_rgb.shape[1], o_rgb.shape[0]))
        w = o_rgb.shape[1]
        split = np.hstack([o_rgb[:, :w//2], p_resized[:, w//2:]])
        st.subheader('Split Comparison (left: original, right: processed)')
        st.image(to_pil(split), use_column_width=True)
    except Exception as e:
        st.error('Split comparison error: ' + str(e))

# Histograms
if show_histograms and st.session_state.get('orig', None) is not None:
    o = st.session_state['orig']
    p = st.session_state.get('proc', o)
    fig, axs = plt.subplots(2,1,figsize=(6,4))
    if o.ndim==3:
        for i, col in enumerate(['b','g','r']):
            axs[0].hist(o[:,:,i].ravel(), bins=256, alpha=0.4, label=col)
    else:
        axs[0].hist(o.ravel(), bins=256)
    axs[0].set_title('Original Histogram')
    if p is not None:
        if p.ndim==3:
            for i, col in enumerate(['b','g','r']):
                axs[1].hist(p[:,:,i].ravel(), bins=256, alpha=0.4, label=col)
        else:
            axs[1].hist(p.ravel(), bins=256)
        axs[1].set_title('Processed Histogram')
    st.pyplot(fig)

# Status bar at bottom
st.markdown('---')
cols = st.columns(4)
orig = st.session_state.get('orig', None)
proc = st.session_state.get('proc', None)
if orig is not None:
    with cols[0]:
        st.write(f"Dimensions (H x W x C): {orig.shape}")
    with cols[1]:
        # DPI not reliably stored; attempt to read from PIL info
        try:
            dpi = st.session_state.get('orig_pil').info.get('dpi', None)
        except Exception:
            dpi = None
        st.write(f"DPI: {dpi if dpi else 'N/A'}")
    with cols[2]:
        try:
            size = len(pil_bytes(to_pil(orig), fmt='PNG'))
        except Exception:
            size = 'N/A'
        st.write(f"Original (est) size: {size} bytes")
    with cols[3]:
        st.write(f"File format: {save_format if uploaded is not None else 'N/A'}")

# Download / Save processed image
if proc is not None:
    pil_proc = to_pil(proc)
    bts = pil_bytes(pil_proc, fmt=save_format, quality=save_quality)
    st.download_button('Download processed image', data=bts, file_name=f'processed.{save_format.lower()}', mime=f'image/{save_format.lower()}')
    if st.session_state.get('_save_requested', False):
        # write to local file in app folder
        fname = f'processed.{save_format.lower()}'
        with open(fname, 'wb') as f:
            f.write(bts)
        st.success(f'Saved processed image as {fname}')
        st.session_state['_save_requested'] = False

# End
st.caption('Toolkit created for Assignment 3 — Image Processing Fundamentals. Fork & submit under /assignments/Task-3/.')
