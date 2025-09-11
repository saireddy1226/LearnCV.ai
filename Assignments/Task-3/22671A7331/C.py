# Image Processing & Analysis Toolkit (Streamlit + OpenCV)
# Filename: app.py
# Requirements: streamlit, opencv-python, numpy, pillow
# Install: pip install streamlit opencv-python numpy pillow

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime

st.set_page_config(page_title="Image Processing & Analysis Toolkit", layout='wide')

# --------------------------- Utility functions ---------------------------

def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert('RGB')
    return np.array(image)


def np_to_pil(img: np.ndarray):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img
    return Image.fromarray(img)


def pil_to_np(img: Image.Image):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def get_image_info(img_array, uploaded_file=None):
    h, w = img_array.shape[:2]
    channels = 1 if img_array.ndim == 2 else img_array.shape[2]
    file_format = None
    file_size = None
    dpi = None
    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            file_size = uploaded_file.size
            file_format = uploaded_file.type
        except Exception:
            pass
    return {
        'width': w,
        'height': h,
        'channels': channels,
        'format': file_format,
        'filesize': file_size,
        'dpi': dpi
    }


def safe_show(img):
    if img is None:
        return None
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            return Image.fromarray(img)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
    elif isinstance(img, Image.Image):
        return img
    else:
        raise ValueError('Unsupported image type')

# --------------------------- Core operations ---------------------------

def convert_color(img, mode):
    # img is BGR numpy
    if mode == 'BGR->RGB':
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if mode == 'BGR->GRAY':
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == 'BGR->HSV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if mode == 'BGR->YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if mode == 'RGB->BGR':
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def rotate_image(img, angle, center=None, scale=1.0):
    h, w = img.shape[:2]
    if center is None:
        center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(img, M, (w, h))


def translate_image(img, tx, ty):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h))


def scale_image(img, fx, fy):
    return cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)


def affine_transform(img, src_pts, dst_pts):
    M = cv2.getAffineTransform(np.float32(src_pts), np.float32(dst_pts))
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h))


def perspective_transform(img, src_pts, dst_pts):
    M = cv2.getPerspectiveTransform(np.float32(src_pts), np.float32(dst_pts))
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, M, (w, h))


def apply_filter(img, filt, ksize=3):
    if filt == 'gaussian':
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    if filt == 'median':
        return cv2.medianBlur(img, ksize)
    if filt == 'mean':
        return cv2.blur(img, (ksize, ksize))
    return img


def edge_detection(img, method='canny', **kwargs):
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if method == 'canny':
        th1 = kwargs.get('th1', 100)
        th2 = kwargs.get('th2', 200)
        return cv2.Canny(gray, th1, th2)
    if method == 'sobel':
        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kwargs.get('ksize', 3))
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kwargs.get('ksize', 3))
        mag = np.sqrt(dx**2 + dy**2)
        mag = np.uint8(np.clip(mag / mag.max() * 255, 0, 255))
        return mag
    if method == 'laplacian':
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap = np.uint8(np.clip(np.abs(lap), 0, 255))
        return lap
    return gray


def morphological(img, op='dilate', kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if op == 'dilate':
        return cv2.dilate(img, kernel, iterations=iterations)
    if op == 'erode':
        return cv2.erode(img, kernel, iterations=iterations)
    if op == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    if op == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img


def histogram_equalization(img):
    if img.ndim == 2:
        return cv2.equalizeHist(img)
    # convert to YCrCb and equalize Y channel
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def contrast_stretch(img):
    in_min = np.percentile(img, 2)
    in_max = np.percentile(img, 98)
    img_stretched = np.clip((img - in_min) * 255.0 / (in_max - in_min), 0, 255).astype(np.uint8)
    return img_stretched


def sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def save_image(img, filename, fmt='PNG', quality=95):
    pil_img = safe_show(img)
    buf = io.BytesIO()
    params = {}
    if fmt.upper() == 'JPEG':
        pil_img.save(buf, format='JPEG', quality=quality)
    else:
        pil_img.save(buf, format=fmt)
    buf.seek(0)
    with open(filename, 'wb') as f:
        f.write(buf.read())
    return os.path.getsize(filename)

# --------------------------- Streamlit App Layout ---------------------------

st.title('📸 Image Processing & Analysis Toolkit')

# Top menu simulation
menu = st.sidebar.selectbox('Menu', ['File', 'Operations', 'Help'])

uploaded_file = st.sidebar.file_uploader('Open Image', type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'])

# Sidebar operation categories
st.sidebar.markdown('---')
mode = st.sidebar.radio('Mode', ['Single Image', 'Webcam (Realtime)'])

# Operation categories inside expanders
info_exp = st.sidebar.expander('Image Info', expanded=False)
color_exp = st.sidebar.expander('Color Conversions', expanded=False)
trans_exp = st.sidebar.expander('Transformations', expanded=False)
filter_exp = st.sidebar.expander('Filtering & Morphology', expanded=False)
enh_exp = st.sidebar.expander('Enhancement', expanded=False)
edge_exp = st.sidebar.expander('Edge Detection', expanded=False)
compress_exp = st.sidebar.expander('Compression', expanded=False)

# placeholders for processed image and original
orig_placeholder = st.empty()
proc_placeholder = st.empty()

# initialize session state for images
if 'orig' not in st.session_state:
    st.session_state.orig = None
if 'proc' not in st.session_state:
    st.session_state.proc = None

# Load image
if uploaded_file is not None and mode == 'Single Image':
    try:
        img_np = load_image(uploaded_file)  # RGB
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        st.session_state.orig = img_bgr
        st.session_state.proc = img_bgr.copy()
    except Exception as e:
        st.sidebar.error(f'Error loading image: {e}')

# Webcam mode
if mode == 'Webcam (Realtime)':
    run = st.checkbox('Run Webcam')
    FRAME_PLACE = st.empty()
    cap = None
    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error('Could not open webcam')
            run = False
    if run and cap is not None:
        stframe = st.empty()
        while run:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels='RGB')
            if not st.session_state.get('webcam_stop', False):
                pass
            run = st.button('Stop Webcam')
        cap.release()

# --------------------------- Image Info ---------------------------
with info_exp:
    if st.session_state.orig is not None:
        info = get_image_info(st.session_state.orig, uploaded_file)
        st.write('**Dimensions:**', f"{info['width']} x {info['height']}")
        st.write('**Channels:**', info['channels'])
        st.write('**File format:**', info['format'])
        st.write('**File size (bytes):**', info['filesize'])

# --------------------------- Color conversions ---------------------------
with color_exp:
    conv = st.selectbox('Choose conversion', ['None', 'BGR->RGB', 'BGR->GRAY', 'BGR->HSV', 'BGR->YCrCb'])
    if conv != 'None' and st.session_state.orig is not None:
        st.session_state.proc = convert_color(st.session_state.orig, conv)

# --------------------------- Transformations ---------------------------
with trans_exp:
    trans = st.selectbox('Transformation', ['None', 'Rotate', 'Scale', 'Translate', 'Affine', 'Perspective'])
    if st.session_state.orig is not None:
        if trans == 'Rotate':
            angle = st.slider('Angle', -180, 180, 0)
            scale = st.slider('Scale', 0.1, 3.0, 1.0)
            st.session_state.proc = rotate_image(st.session_state.orig, angle, scale=scale)
        elif trans == 'Scale':
            fx = st.slider('Scale X (fx)', 0.1, 3.0, 1.0)
            fy = st.slider('Scale Y (fy)', 0.1, 3.0, 1.0)
            st.session_state.proc = scale_image(st.session_state.orig, fx, fy)
        elif trans == 'Translate':
            tx = st.slider('Translate X (px)', -200, 200, 0)
            ty = st.slider('Translate Y (px)', -200, 200, 0)
            st.session_state.proc = translate_image(st.session_state.orig, tx, ty)
        elif trans == 'Affine':
            st.write('Affine transform sliders are normalized to image size')
            h, w = st.session_state.orig.shape[:2]
            src = np.float32([[0,0],[w-1,0],[0,h-1]])
            dst = np.float32([
                [st.slider('dst x0', 0, w, 0), st.slider('dst y0', 0, h, 0)],
                [st.slider('dst x1', 0, w, w-1), st.slider('dst y1', 0, h, 0)],
                [st.slider('dst x2', 0, w, 0), st.slider('dst y2', 0, h, h-1)]
            ])
            st.session_state.proc = affine_transform(st.session_state.orig, src, dst)
        elif trans == 'Perspective':
            h, w = st.session_state.orig.shape[:2]
            st.write('Adjust destination corner positions')
            src_pts = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
            dst_pts = np.float32([
                [st.slider('x0', 0, w, 0), st.slider('y0', 0, h, 0)],
                [st.slider('x1', 0, w, w-1), st.slider('y1', 0, h, 0)],
                [st.slider('x2', 0, w, w-1), st.slider('y2', 0, h, h-1)],
                [st.slider('x3', 0, w, 0), st.slider('y3', 0, h, h-1)]
            ])
            st.session_state.proc = perspective_transform(st.session_state.orig, src_pts, dst_pts)

# --------------------------- Filtering & Morphology ---------------------------
with filter_exp:
    filt = st.selectbox('Filter', ['None', 'Gaussian', 'Median', 'Mean'])
    ksize = st.slider('Kernel size (odd)', 1, 31, 3, step=2)
    if filt != 'None' and st.session_state.orig is not None:
        st.session_state.proc = apply_filter(st.session_state.orig, filt.lower(), ksize=ksize)

    morph = st.selectbox('Morphology', ['None', 'Dilate', 'Erode', 'Opening', 'Closing'])
    m_ksize = st.slider('Morph kernel size', 1, 31, 3, step=1)
    m_iter = st.slider('Iterations', 1, 10, 1)
    if morph != 'None' and st.session_state.orig is not None:
        st.session_state.proc = morphological(st.session_state.orig, morph.lower(), kernel_size=m_ksize, iterations=m_iter)

# --------------------------- Enhancement ---------------------------
with enh_exp:
    enh = st.selectbox('Enhancement', ['None', 'Histogram Equalization', 'Contrast Stretching', 'Sharpen'])
    if enh != 'None' and st.session_state.orig is not None:
        if enh == 'Histogram Equalization':
            st.session_state.proc = histogram_equalization(st.session_state.orig)
        elif enh == 'Contrast Stretching':
            st.session_state.proc = contrast_stretch(st.session_state.orig)
        elif enh == 'Sharpen':
            st.session_state.proc = sharpen(st.session_state.orig)

# --------------------------- Edge Detection ---------------------------
with edge_exp:
    edge = st.selectbox('Edge Method', ['None', 'Canny', 'Sobel', 'Laplacian'])
    if edge != 'None' and st.session_state.orig is not None:
        if edge == 'Canny':
            t1 = st.slider('Threshold 1', 0, 500, 100)
            t2 = st.slider('Threshold 2', 0, 500, 200)
            st.session_state.proc = edge_detection(st.session_state.orig, 'canny', th1=t1, th2=t2)
        elif edge == 'Sobel':
            k = st.slider('Sobel ksize', 1, 7, 3, step=2)
            st.session_state.proc = edge_detection(st.session_state.orig, 'sobel', ksize=k)
        elif edge == 'Laplacian':
            st.session_state.proc = edge_detection(st.session_state.orig, 'laplacian')

# --------------------------- Compression ---------------------------
with compress_exp:
    save_fmt = st.selectbox('Save format', ['PNG', 'JPEG', 'BMP'])
    if st.button('Save Processed Image') and st.session_state.proc is not None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f'processed_{timestamp}.{save_fmt.lower()}'
        try:
            size = save_image(st.session_state.proc, fname, fmt=save_fmt, quality=90)
            st.success(f'Saved {fname} ({size} bytes)')
            with open(fname, 'rb') as f:
                st.download_button('Download file', f, file_name=fname)
        except Exception as e:
            st.error(f'Error saving file: {e}')

# --------------------------- Comparison & Display ---------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader('Original Image')
    if st.session_state.orig is not None:
        st.image(cv2.cvtColor(st.session_state.orig, cv2.COLOR_BGR2RGB), use_column_width=True)
    else:
        st.info('Upload an image to begin')
with col2:
    st.subheader('Processed Image')
    if st.session_state.proc is not None:
        proc_vis = safe_show(st.session_state.proc)
        st.image(proc_vis, use_column_width=True)

# Status bar
st.markdown('---')
if st.session_state.orig is not None:
    st.write('**Status:**')
    info = get_image_info(st.session_state.orig, uploaded_file)
    st.write(f"Dimensions: {info['width']} x {info['height']} | Channels: {info['channels']} | File size: {info['filesize']}")

st.caption('Designed for educational purposes — Image Processing Toolkit with Streamlit & OpenCV')
