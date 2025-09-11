# Streamlit Image Toolkit - app.py
# Implements a GUI for image operations: color conversions, transforms, filters, morphology,
# enhancement, edge detection, compression and save/download.

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ExifTags
import io
import os
import tempfile


st.set_page_config(page_title="Image Toolkit", layout="wide")

# ----------------- Utility functions -----------------

def read_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    return img


def cv2_to_pil(img):
    if img is None:
        return None
    
    # If grayscale
    if len(img.shape) == 2:
        return Image.fromarray(img)
    
    # If color image
    channels = img.shape[2]
    if channels == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
    elif channels == 4:
        img_rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        return Image.fromarray(img_rgba)
    else:
        # Fallback to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if channels > 1 else img
        return Image.fromarray(gray)



def pil_to_cv2(pil_img):
    arr = np.array(pil_img)
    if arr.ndim == 2:
        return arr
    if arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    return arr




def get_image_info(img, uploaded_name=None):
    if img is None:
        return {}
    
    h, w = img.shape[:2]
    c = 1 if img.ndim == 2 else img.shape[2]
    info = {
        'Height': h,
        'Width': w,
        'Channels': c,
        'Shape': img.shape,
        'DPI': 'Unknown',
        'Filename': uploaded_name or 'N/A'
    }
    return info


def to_bytes(img, fmt='PNG'):
    pil = cv2_to_pil(img)
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return buf.getvalue()


# ----------------- Image operation implementations -----------------

def convert_color(img, op):
    if img is None:
        return None
    if op == 'RGB -> BGR':
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[-1] == 3 else img
    if op == 'BGR -> RGB':
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[-1] == 3 else img
    if op == 'RGB -> HSV' or op == 'BGR -> HSV':
        # OpenCV assumes BGR input
        if img.shape[-1] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return img
    if op == 'RGB -> YCrCb' or op == 'BGR -> YCrCb':
        if img.shape[-1] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        return img
    if op == 'Grayscale':
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def rotate_image(img, angle, center=None, scale=1.0):
    h, w = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    out = cv2.warpAffine(img, M, (w, h))
    return out


def scale_image(img, fx, fy):
    return cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)


def translate_image(img, tx, ty):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h))


def affine_transform(img, src_pts, dst_pts):
    M = cv2.getAffineTransform(np.float32(src_pts), np.float32(dst_pts))
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h))


def perspective_transform(img, src_pts, dst_pts):
    M = cv2.getPerspectiveTransform(np.float32(src_pts), np.float32(dst_pts))
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, M, (w, h))


def apply_filter(img, filter_name, ksize=3):
    if filter_name == 'Gaussian':
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    if filter_name == 'Mean':
        return cv2.blur(img, (ksize, ksize))
    if filter_name == 'Median':
        return cv2.medianBlur(img, ksize)
    if filter_name == 'Sobel':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        mag = cv2.magnitude(sx, sy)
        mag = np.uint8(np.clip(mag, 0, 255))
        return mag
    if filter_name == 'Laplacian':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap = np.uint8(np.clip(np.abs(lap), 0, 255))
        return lap
    return img


def morphology(img, op, ksize=3, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    if op == 'Dilation':
        return cv2.dilate(img, kernel, iterations=iterations)
    if op == 'Erosion':
        return cv2.erode(img, kernel, iterations=iterations)
    if op == 'Opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    if op == 'Closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img


def enhance(img, method):
    if method == 'Histogram Equalization':
        if len(img.shape) == 3:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            return cv2.equalizeHist(img)
    if method == 'Sharpen':
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)
    if method == 'Contrast Stretch':
        in_min = np.min(img)
        in_max = np.max(img)
        out = (img - in_min) * (255.0 / (in_max - in_min + 1e-8))
        return np.uint8(out)
    return img


def edge_detection(img, method, thresh1=100, thresh2=200):
    if method == 'Canny':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, thresh1, thresh2)
    if method == 'Sobel':
        return apply_filter(img, 'Sobel')
    if method == 'Laplacian':
        return apply_filter(img, 'Laplacian')
    return img


# ----------------- UI Layout -----------------

# Top menu bar simulated via columns (Streamlit doesn't provide native menu bars yet)
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    st.markdown("**File**")
with col2:
    st.markdown("### Image Toolkit")
with col3:
    if st.button('Exit'):
        st.stop()

# Sidebar controls
st.sidebar.title('Operations')
uploaded_file = st.sidebar.file_uploader('Open → Upload an image', type=['png','jpg','jpeg','bmp','tiff'])

# Sidebar sections
op_category = st.sidebar.selectbox('Category', [
    'Image Info', 'Color Conversions', 'Transformations', 'Filtering & Morphology',
    'Enhancement', 'Edge Detection', 'Compression & Save'
])

# Default placeholders
original_img = None
processed_img = None
uploaded_name = None

if uploaded_file is not None:
    uploaded_name = uploaded_file.name
    original_img = read_image(uploaded_file)
    processed_img = original_img.copy()

if original_img is not None:
    pil_img = cv2_to_pil(original_img)
    if pil_img is not None:
        st.image(pil_img, use_container_width=True)
    else:
        st.info('Cannot display this image.')
else:
    st.info('No image uploaded yet.')

# Parameters & operation selectors per category
if op_category == 'Image Info':
    st.sidebar.markdown('---')
    st.sidebar.write('Image fundamentals and metadata')

elif op_category == 'Color Conversions':
    choice = st.sidebar.selectbox('Convert', ['Grayscale', 'BGR -> RGB', 'RGB -> BGR', 'BGR -> HSV', 'BGR -> YCrCb'])
    if st.sidebar.button('Apply Color Conversion'):
        processed_img = convert_color(processed_img, choice)

elif op_category == 'Transformations':
    trans = st.sidebar.selectbox('Transform', ['Rotate', 'Scale', 'Translate', 'Affine', 'Perspective'])
    if trans == 'Rotate':
        angle = st.sidebar.slider('Angle', -180, 180, 0)
        scale = st.sidebar.slider('Scale', 0.1, 3.0, 1.0)
        if st.sidebar.button('Apply Rotate'):
            processed_img = rotate_image(processed_img, angle, scale=scale)
    elif trans == 'Scale':
        fx = st.sidebar.slider('Scale X (fx)', 0.1, 3.0, 1.0)
        fy = st.sidebar.slider('Scale Y (fy)', 0.1, 3.0, 1.0)
        if st.sidebar.button('Apply Scale'):
            processed_img = scale_image(processed_img, fx, fy)
    elif trans == 'Translate':
        tx = st.sidebar.slider('Translate X', -500, 500, 0)
        ty = st.sidebar.slider('Translate Y', -500, 500, 0)
        if st.sidebar.button('Apply Translate'):
            processed_img = translate_image(processed_img, tx, ty)
    elif trans == 'Affine':
        st.sidebar.write('Affine uses three source and destination points')
        if st.sidebar.button('Apply Simple Affine (center shear)'):
            h, w = processed_img.shape[:2]
            src = [[0,0],[w-1,0],[0,h-1]]
            dst = [[0,h*0.33],[w*0.85, h*0.25],[w*0.15,h*0.7]]
            processed_img = affine_transform(processed_img, src, dst)
    elif trans == 'Perspective':
        if st.sidebar.button('Apply Perspective'):
            h, w = processed_img.shape[:2]
            src = [[0,0],[w-1,0],[w-1,h-1],[0,h-1]]
            dst = [[w*0.0,h*0.33],[w*0.85,0],[w*0.9,h*0.9],[w*0.2,h*0.9]]
            processed_img = perspective_transform(processed_img, src, dst)

elif op_category == 'Filtering & Morphology':
    choice = st.sidebar.selectbox('Filter/Morph', ['Gaussian','Mean','Median','Sobel','Laplacian','Dilation','Erosion','Opening','Closing'])
    k = st.sidebar.slider('Kernel size', 1, 31, 3, step=2)
    iters = st.sidebar.slider('Iterations (morphology)', 1, 10, 1)
    if st.sidebar.button('Apply'):
        if choice in ['Gaussian','Mean','Median','Sobel','Laplacian']:
            processed_img = apply_filter(processed_img, choice, ksize=k)
        else:
            processed_img = morphology(processed_img, choice, ksize=k, iterations=iters)

elif op_category == 'Enhancement':
    choice = st.sidebar.selectbox('Enhance', ['Histogram Equalization','Sharpen','Contrast Stretch'])
    if st.sidebar.button('Apply Enhance'):
        processed_img = enhance(processed_img, choice)

elif op_category == 'Edge Detection':
    method = st.sidebar.selectbox('Method', ['Canny','Sobel','Laplacian'])
    t1 = st.sidebar.slider('Canny Thresh1', 0, 500, 100)
    t2 = st.sidebar.slider('Canny Thresh2', 0, 500, 200)
    if st.sidebar.button('Apply Edge Detection'):
        processed_img = edge_detection(processed_img, method, t1, t2)

elif op_category == 'Compression & Save':
    fmt = st.sidebar.selectbox('Save format', ['PNG','JPG','BMP'])
    quality = st.sidebar.slider('JPEG Quality (if JPG)', 10, 100, 90)
    if st.sidebar.button('Save Processed Image'):
        b = to_bytes(processed_img, fmt=fmt)
        st.sidebar.download_button('Download', data=b, file_name=f'processed.{fmt.lower()}', mime=f'image/{fmt.lower()}')

# Comparison mode
compare = st.sidebar.checkbox('Comparison mode (split view)', value=False)

# Display area: two columns for original and processed
left_col, right_col = st.columns(2)

# --- Updated Streamlit Image Toolkit - app.py ---
# Fix deprecated use_column_width warning by replacing with use_container_width

# In the display area, replace all st.image calls:

with left_col:
    st.subheader('Original Image')
    if original_img is not None:
        st.image(cv2_to_pil(original_img), use_container_width=True)
    else:
        st.info('No image uploaded yet.')

with right_col:
    st.subheader('Processed Image')
    if processed_img is not None:
        if compare and original_img is not None:
            h = min(original_img.shape[0], processed_img.shape[0])
            w = min(original_img.shape[1], processed_img.shape[1])
            left_half = original_img[:h, :w]
            right_half = processed_img[:h, :w]
            if left_half.ndim == 2:
                left_half = cv2.cvtColor(left_half, cv2.COLOR_GRAY2BGR)
            if right_half.ndim == 2:
                right_half = cv2.cvtColor(right_half, cv2.COLOR_GRAY2BGR)
            split = np.concatenate([left_half[:, :w//2], right_half[:, w//2:]], axis=1)
            st.image(cv2_to_pil(split), use_container_width=True)
        else:
            st.image(cv2_to_pil(processed_img), use_container_width=True)
    else:
        st.info('No processed image available.')


# Status bar - show properties dynamically
st.markdown('---')
if processed_img is not None:
    info = get_image_info(processed_img, uploaded_name)
    size_bytes = len(to_bytes(processed_img, fmt='PNG'))
    st.caption(f"Dimensions: {info.get('Height')} x {info.get('Width')} | Channels: {info.get('Channels')} | File: {info.get('Filename')} | Estimated size (PNG): {size_bytes} bytes")
else:
    st.caption('No image loaded')

# Footer actions

col_a, col_b, col_c = st.columns([1,1,1])
with col_a:
    if st.button('Reset to Original') and original_img is not None:
        processed_img = original_img.copy()
        st.experimental_rerun()   # <-- THIS LINE causes the error

with col_b:
    if st.button('Save as PNG') and processed_img is not None:
        b = to_bytes(processed_img, fmt='PNG')
        st.download_button('Download PNG', data=b, file_name='processed.png', mime='image/png')
with col_c:
    if st.button('Save as JPG') and processed_img is not None:
        b = to_bytes(processed_img, fmt='JPEG')
        st.download_button('Download JPG', data=b, file_name='processed.jpg', mime='image/jpeg')

# End of app
# Requirement.txt  
#streamlit
#opencv-python-headless
# numpy
# pillow
# matplotlib
