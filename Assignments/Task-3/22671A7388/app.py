"""
Streamlit Image Processing Toolkit
Filename: app.py
A single-file Streamlit app that demonstrates image processing operations using OpenCV.
Supports: upload, save, color conversions, transforms, filters, morphology,
edge detection, compression comparison, webcam mode, split view, sliders.

Usage:
$ pip install streamlit opencv-python-headless numpy matplotlib pillow
$ streamlit run app.py

Note: On some platforms use opencv-python instead of headless if you need GUI support.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import os
import tempfile
import math

st.set_page_config(page_title="Image Processing Toolkit", layout="wide")

# ----------------- Helper functions -----------------

def read_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    return img


def cv2_to_pil(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return Image.fromarray(img)
    # Convert BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def pil_to_cv2(pil_img):
    arr = np.array(pil_img)
    if arr.ndim == 2:
        return arr
    # PIL uses RGB
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def image_info(img, filename=None):
    info = {}
    if img is None:
        return info
    info['shape'] = img.shape
    h, w = img.shape[:2]
    info['dimensions'] = (int(h), int(w))
    info['channels'] = 1 if img.ndim == 2 else img.shape[2]
    if filename:
        try:
            info['file_size_bytes'] = os.path.getsize(filename)
        except Exception:
            info['file_size_bytes'] = None
    return info


def ensure_3ch(img):
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        # drop alpha
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


# Color conversions using OpenCV

def convert_color(img, mode):
    if img is None:
        return None
    img = ensure_3ch(img)
    if mode == 'RGB -> BGR':
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if mode == 'BGR -> RGB':
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if mode == 'RGB -> HSV' or mode == 'BGR -> HSV':
        # OpenCV expects BGR
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if mode == 'HSV -> BGR' or mode == 'HSV -> RGB':
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    if mode == 'RGB -> YCrCb' or mode == 'BGR -> YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if mode == 'YCrCb -> BGR':
        return cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    if mode == 'Grayscale':
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# Transformations

def rotate_image(img, angle, keep_size=True):
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    if keep_size:
        rotated = cv2.warpAffine(img, M, (w, h))
    else:
        # compute new bounding
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        rotated = cv2.warpAffine(img, M, (new_w, new_h))
    return rotated


def scale_image(img, fx, fy):
    return cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)


def translate_image(img, tx, ty):
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h))


def affine_transform(img, pts1, pts2):
    M = cv2.getAffineTransform(np.float32(pts1), np.float32(pts2))
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h))


def perspective_transform(img, src_pts, dst_pts):
    M = cv2.getPerspectiveTransform(np.float32(src_pts), np.float32(dst_pts))
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, M, (w, h))


# Filters & Morphology

def apply_filter(img, filter_name, ksize=3):
    if img is None:
        return None
    if filter_name == 'Gaussian':
        k = max(1, int(ksize))
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(img, (k, k), 0)
    if filter_name == 'Median':
        k = max(1, int(ksize))
        if k % 2 == 0:
            k += 1
        return cv2.medianBlur(img, k)
    if filter_name == 'Mean':
        k = max(1, int(ksize))
        return cv2.blur(img, (k, k))
    return img


def edge_filter(img, method='Sobel'):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if method == 'Sobel':
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag = np.uint8(np.clip(mag, 0, 255))
        return mag
    if method == 'Laplacian':
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap = np.uint8(np.clip(np.absolute(lap), 0, 255))
        return lap
    return gray


def morphology(img, op='dilate', kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if op == 'dilate':
        return cv2.dilate(img, kernel, iterations=iterations)
    if op == 'erode':
        return cv2.erode(img, kernel, iterations=iterations)
    if op == 'open':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    if op == 'close':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img


# Enhancement

def histogram_equalization(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.equalizeHist(img)
    # convert to YCrCb and equalize Y
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)


def contrast_stretch(img):
    # simple linear contrast stretch based on percentiles
    if img is None:
        return None
    in_min = np.percentile(img, 2)
    in_max = np.percentile(img, 98)
    out = (img - in_min) * (255.0 / (in_max - in_min + 1e-5))
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


# Compression utilities

def save_image_bytes(img, fmt='PNG', quality=95):
    pil = cv2_to_pil(img)
    buf = io.BytesIO()
    save_kwargs = {}
    if fmt.upper() in ['JPG', 'JPEG']:
        save_kwargs['quality'] = int(quality)
    pil.save(buf, format=fmt, **save_kwargs)
    data = buf.getvalue()
    return data


# Bitwise ops

def bitwise_op(img1, img2, op='and'):
    a = ensure_3ch(img1)
    b = ensure_3ch(img2)
    # make same size
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a = a[:h, :w]
    b = b[:h, :w]
    if op == 'and':
        return cv2.bitwise_and(a, b)
    if op == 'or':
        return cv2.bitwise_or(a, b)
    if op == 'xor':
        return cv2.bitwise_xor(a, b)
    if op == 'not':
        return cv2.bitwise_not(a)
    return a


# ----------------- Streamlit UI -----------------

st.title("🖼️ Image Processing Toolkit — Streamlit + OpenCV")

# Top menu imitation
menu = st.selectbox("Menu", ['File', 'Help'])
if menu == 'Help':
    st.info('This app demonstrates common image-processing operations. Use the sidebar to choose operations.')

# Layout: left sidebar for controls, main area for images
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Open → Upload an image", type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'])
    save_fmt = st.selectbox("Save format", ['PNG', 'JPG', 'BMP'])
    # Options
    mode = st.radio("Mode", ['Single Image', 'Compare Two Images', 'Realtime Video (Webcam)'])

    st.markdown("---")
    st.subheader("Image Info")
    show_info = st.checkbox('Show image info', value=True)

    st.markdown("---")
    st.subheader("Operations")
    category = st.selectbox('Category', [
        'Color Conversions', 'Transformations', 'Filters & Morphology',
        'Enhancement', 'Edge Detection', 'Bitwise Ops', 'Compression'
    ])

    # Dynamic options per category
    if category == 'Color Conversions':
        color_op = st.selectbox('Conversion', ['Grayscale', 'BGR -> RGB', 'RGB -> BGR', 'BGR -> HSV', 'HSV -> BGR', 'BGR -> YCrCb', 'YCrCb -> BGR'])
    elif category == 'Transformations':
        trans_op = st.selectbox('Transformation', ['Rotate', 'Scale', 'Translate', 'Affine', 'Perspective'])
        rot_angle = st.slider('Rotation angle', -360, 360, 0)
        keep_size = st.checkbox('Keep size when rotating', True)
        scale_fx = st.slider('Scale fx', 0.1, 3.0, 1.0)
        scale_fy = st.slider('Scale fy', 0.1, 3.0, 1.0)
        tx = st.slider('Translate X (pixels)', -500, 500, 0)
        ty = st.slider('Translate Y (pixels)', -500, 500, 0)
    elif category == 'Filters & Morphology':
        filt = st.selectbox('Filter', ['None', 'Gaussian', 'Median', 'Mean'])
        ksize = st.slider('Kernel size', 1, 31, 3)
        morph = st.selectbox('Morphology', ['None', 'dilate', 'erode', 'open', 'close'])
        morph_iter = st.slider('Morph iterations', 1, 10, 1)
    elif category == 'Enhancement':
        enh = st.selectbox('Enhancement', ['Histogram Equalization', 'Contrast Stretching', 'Sharpen'])
    elif category == 'Edge Detection':
        edge_method = st.selectbox('Edge Method', ['Canny', 'Sobel', 'Laplacian'])
        if edge_method == 'Canny':
            canny_t1 = st.slider('Canny threshold1', 0, 500, 100)
            canny_t2 = st.slider('Canny threshold2', 0, 500, 200)
    elif category == 'Bitwise Ops':
        bw_mode = st.selectbox('Bitwise', ['and', 'or', 'xor', 'not'])
        second_file = st.file_uploader('Upload second image (for bitwise)', type=['png', 'jpg', 'jpeg', 'bmp'], key='bitwise2')
    elif category == 'Compression':
        comp_quality = st.slider('JPEG Quality', 10, 100, 95)
        compare_sizes = st.checkbox('Compare file sizes (PNG vs JPG)', True)

    st.markdown('---')
    st.write('Preview & Actions:')
    save_button = st.button('Save Processed Image')
    st.write('')

# Main area: two columns for original and processed
col1, col2 = st.columns(2)
orig_placeholder = col1.empty()
proc_placeholder = col2.empty()

# Status bar (bottom)
status = st.container()

# Load image(s)
img = None
img2 = None
img_filename = None

if uploaded_file is not None:
    try:
        uploaded_file.seek(0)
        img = read_image(uploaded_file)
        img_filename = getattr(uploaded_file, 'name', None)
    except Exception as e:
        st.error(f"Failed to read image: {e}")

if category == 'Bitwise Ops' and 'second_file' in locals() and second_file is not None:
    try:
        second_file.seek(0)
        img2 = read_image(second_file)
    except Exception:
        img2 = None

# Realtime video handling
if mode == 'Realtime Video (Webcam)':
    st.sidebar.write('Webcam Mode')
    run_cam = st.sidebar.button('Start Webcam')
    stop_cam = st.sidebar.button('Stop Webcam')
    cam_placeholder = st.empty()
    if run_cam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error('Could not open webcam')
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                display_frame = frame.copy()
                # apply selected operations in realtime
                if category == 'Filters & Morphology' and filt != 'None':
                    display_frame = apply_filter(display_frame, filt, ksize)
                if category == 'Edge Detection' and edge_method == 'Canny':
                    gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, canny_t1, canny_t2)
                    display_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                cam_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels='RGB')
                # stop condition: Streamlit will rerun and exit loop when user presses stop
                if stop_cam:
                    break
            cap.release()
    st.stop()

# If compare mode, ask for second upload
if mode == 'Compare Two Images' and img is not None and img2 is None:
    # ask for second upload
    img2_file = st.sidebar.file_uploader('Upload second image to compare', type=['png', 'jpg', 'jpeg', 'bmp'], key='compare2')
    if img2_file is not None:
        img2 = read_image(img2_file)

# Show original
if img is not None:
    try:
        orig_placeholder.image(cv2_to_pil(img), caption='Original Image')
    except Exception:
        orig_placeholder.write('Unable to display original')
else:
    orig_placeholder.write('No image uploaded yet')

# Process image according to category
processed = None
if img is not None:
    # start with a copy
    processed = img.copy()

    if category == 'Color Conversions':
        processed = convert_color(processed, color_op)
    elif category == 'Transformations':
        if trans_op == 'Rotate':
            processed = rotate_image(processed, rot_angle, keep_size)
        elif trans_op == 'Scale':
            processed = scale_image(processed, scale_fx, scale_fy)
        elif trans_op == 'Translate':
            processed = translate_image(processed, tx, ty)
        elif trans_op == 'Affine':
            h, w = processed.shape[:2]
            # default example: small shear
            pts1 = [[0,0],[w-1,0],[0,h-1]]
            pts2 = [[0,h*0.1],[w*0.9, h*0.05],[w*0.05,h*0.9]]
            processed = affine_transform(processed, pts1, pts2)
        elif trans_op == 'Perspective':
            h, w = processed.shape[:2]
            src = [[0,0],[w-1,0],[w-1,h-1],[0,h-1]]
            dst = [[w*0.0,h*0.33],[w*0.9,0],[w*0.8,h*0.9],[w*0.2,h*0.8]]
            processed = perspective_transform(processed, src, dst)
    elif category == 'Filters & Morphology':
        if filt != 'None':
            processed = apply_filter(processed, filt, ksize)
        if morph != 'None':
            processed = morphology(processed, morph, kernel_size=ksize, iterations=morph_iter)
    elif category == 'Enhancement':
        if enh == 'Histogram Equalization':
            processed = histogram_equalization(processed)
        elif enh == 'Contrast Stretching':
            processed = contrast_stretch(processed)
        elif enh == 'Sharpen':
            processed = sharpen(processed)
    elif category == 'Edge Detection':
        if edge_method == 'Canny':
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, canny_t1, canny_t2)
            processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif edge_method == 'Sobel':
            edges = edge_filter(processed, 'Sobel')
            processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif edge_method == 'Laplacian':
            edges = edge_filter(processed, 'Laplacian')
            processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif category == 'Bitwise Ops':
        if bw_mode == 'not':
            processed = bitwise_op(processed, processed, op='not')
        else:
            if img2 is None:
                st.warning('Second image required for this operation')
            else:
                processed = bitwise_op(processed, img2, op=bw_mode)
    elif category == 'Compression':
        # no-op — we only handle saving later
        processed = processed

# Compression comparison
comp_info = None
if img is not None and category == 'Compression' and compare_sizes:
    png_bytes = save_image_bytes(img, fmt='PNG')
    jpg_bytes = save_image_bytes(img, fmt='JPEG', quality=comp_quality)
    comp_info = {
        'png_size_kb': len(png_bytes)/1024.0,
        'jpg_size_kb': len(jpg_bytes)/1024.0,
        'jpg_quality': comp_quality
    }

# Show processed
if processed is not None:
    try:
        proc_placeholder.image(cv2_to_pil(processed), caption='Processed Image')
    except Exception:
        proc_placeholder.write('Unable to display processed image')
else:
    proc_placeholder.write('No processed image')

# Split-view bonus
if st.checkbox('Enable split comparison (half original / half processed)') and img is not None and processed is not None:
    def half_half_display(a, b):
        # resize to same size
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a_c = a[:h, :w]
        b_c = b[:h, :w]
        mid = w // 2
        combo = np.concatenate((a_c[:, :mid], b_c[:, mid:]), axis=1)
        return combo
    combo = half_half_display(ensure_3ch(img), ensure_3ch(processed))
    st.image(cv2_to_pil(combo), caption='Split: Left=Original Right=Processed')

# Save processed image
if save_button and processed is not None:
    data = save_image_bytes(processed, fmt=save_fmt, quality=(comp_quality if 'comp_quality' in locals() else 95))
    st.download_button(label='Download processed image', data=data, file_name=f'processed.{save_fmt.lower()}', mime=f'image/{save_fmt.lower()}')

# Show compression info
if comp_info is not None:
    st.write('Compression comparison:')
    st.write(f"PNG size: {comp_info['png_size_kb']:.1f} KB")
    st.write(f"JPEG size (quality={comp_info['jpg_quality']}): {comp_info['jpg_size_kb']:.1f} KB")

# Status bar info
with status:
    if img is not None:
        info = image_info(img, filename=None)
        st.write('---')
        st.markdown('**Status Bar — Image properties**')
        st.write(f"Dimensions (H x W x C): {info.get('shape', '')}")
        st.write(f"Channels: {info.get('channels', '')}")
        if img_filename:
            st.write(f"Filename: {img_filename}")
        if comp_info:
            st.write(f"PNG size: {comp_info['png_size_kb']:.1f} KB — JPG size: {comp_info['jpg_size_kb']:.1f} KB")
    else:
        st.write('---')
        st.write('No image loaded.')

st.markdown('\n---\n')
st.caption('Assignment 3 — Image Processing Toolkit (Streamlit + OpenCV)')
