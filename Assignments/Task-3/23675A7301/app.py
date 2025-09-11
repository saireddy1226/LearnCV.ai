import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance
import io
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Image Processing Toolkit",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .image-container {
        display: flex;
        justify-content: space-around;
        margin-bottom: 2rem;
    }
    .image-box {
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .status-bar {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'image_uploaded' not in st.session_state:
    st.session_state.image_uploaded = False
if 'image_info' not in st.session_state:
    st.session_state.image_info = {}

def get_image_info(image):
    """Extract image information"""
    if image is None:
        return {}
    
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    # Get basic info
    width, height = pil_image.size
    mode = pil_image.mode
    channels = len(pil_image.getbands())
    
    # Try to get DPI info
    dpi = pil_image.info.get('dpi', (72, 72))
    
    # Calculate file size for different formats
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    png_size = len(img_byte_arr.getvalue()) / 1024  # KB
    
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='JPEG', quality=95)
    jpg_size = len(img_byte_arr.getvalue()) / 1024  # KB
    
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='BMP')
    bmp_size = len(img_byte_arr.getvalue()) / 1024  # KB
    
    return {
        'width': width,
        'height': height,
        'channels': channels,
        'mode': mode,
        'dpi': dpi,
        'png_size': png_size,
        'jpg_size': jpg_size,
        'bmp_size': bmp_size
    }

def update_image_info():
    """Update image information in session state"""
    if st.session_state.processed_image is not None:
        st.session_state.image_info = get_image_info(st.session_state.processed_image)
    elif st.session_state.original_image is not None:
        st.session_state.image_info = get_image_info(st.session_state.original_image)

def convert_color(image, conversion):
    """Convert image color space"""
    if image is None:
        return None
    
    # Convert PIL to OpenCV format if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Apply the conversion
    if conversion == 'RGB to BGR':
        converted = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif conversion == 'BGR to RGB':
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif conversion == 'RGB to HSV':
        converted = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif conversion == 'HSV to RGB':
        converted = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    elif conversion == 'RGB to YCbCr':
        converted = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    elif conversion == 'YCbCr to RGB':
        converted = cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)
    elif conversion == 'RGB to Grayscale':
        if len(image.shape) == 3:
            converted = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            converted = image
    else:
        converted = image
    
    return converted

def apply_transformation(image, transformation, **params):
    """Apply geometric transformation to image"""
    if image is None:
        return None
    
    # Convert PIL to OpenCV format if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    height, width = image.shape[:2]
    
    if transformation == 'Rotation':
        angle = params.get('angle', 0)
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        transformed = cv2.warpAffine(image, matrix, (width, height))
    
    elif transformation == 'Scaling':
        scale_x = params.get('scale_x', 1.0)
        scale_y = params.get('scale_y', 1.0)
        transformed = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
    
    elif transformation == 'Translation':
        tx = params.get('tx', 0)
        ty = params.get('ty', 0)
        matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        transformed = cv2.warpAffine(image, matrix, (width, height))
    
    elif transformation == 'Affine Transform':
        # Define points for affine transformation
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
        matrix = cv2.getAffineTransform(pts1, pts2)
        transformed = cv2.warpAffine(image, matrix, (width, height))
    
    elif transformation == 'Perspective Transform':
        # Define points for perspective transformation
        pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
        pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        transformed = cv2.warpPerspective(image, matrix, (300, 300))
    
    else:
        transformed = image
    
    return transformed

def apply_filter(image, filter_type, **params):
    """Apply filter to image"""
    if image is None:
        return None
    
    # Convert PIL to OpenCV format if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    kernel_size = params.get('kernel_size', 5)
    
    if filter_type == 'Gaussian Blur':
        filtered = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    elif filter_type == 'Mean Filter':
        filtered = cv2.blur(image, (kernel_size, kernel_size))
    
    elif filter_type == 'Median Filter':
        filtered = cv2.medianBlur(image, kernel_size)
    
    elif filter_type == 'Sobel':
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
        filtered = cv2.magnitude(sobelx, sobely)
        filtered = np.uint8(np.absolute(filtered))
    
    elif filter_type == 'Laplacian':
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        filtered = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)
        filtered = np.uint8(np.absolute(filtered))
    
    else:
        filtered = image
    
    return filtered

def apply_morphology(image, operation, **params):
    """Apply morphological operation to image"""
    if image is None:
        return None
    
    # Convert PIL to OpenCV format if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    kernel_size = params.get('kernel_size', 5)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    if operation == 'Dilation':
        result = cv2.dilate(gray, kernel, iterations=1)
    elif operation == 'Erosion':
        result = cv2.erode(gray, kernel, iterations=1)
    elif operation == 'Opening':
        result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    elif operation == 'Closing':
        result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    else:
        result = gray
    
    return result

def apply_enhancement(image, enhancement, **params):
    """Apply image enhancement"""
    if image is None:
        return None
    
    # Convert to PIL if it's a numpy array
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    
    if enhancement == 'Histogram Equalization':
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Apply histogram equalization
        enhanced = ImageOps.equalize(image)
    
    elif enhancement == 'Contrast Stretching':
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Apply contrast stretching
        min_val = np.percentile(np.array(image), 2)
        max_val = np.percentile(np.array(image), 98)
        enhanced = ImageOps.autocontrast(image, cutoff=(min_val/255, max_val/255))
    
    elif enhancement == 'Sharpening':
        # Apply sharpening filter
        enhancer = ImageEnhance.Sharpness(image)
        enhanced = enhancer.enhance(2.0)
    
    else:
        enhanced = image
    
    return enhanced

def apply_edge_detection(image, method, **params):
    """Apply edge detection"""
    if image is None:
        return None
    
    # Convert PIL to OpenCV format if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    if method == 'Sobel':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        edges = cv2.magnitude(sobelx, sobely)
        edges = np.uint8(np.absolute(edges))
    
    elif method == 'Canny':
        threshold1 = params.get('threshold1', 100)
        threshold2 = params.get('threshold2', 200)
        edges = cv2.Canny(gray, threshold1, threshold2)
    
    elif method == 'Laplacian':
        edges = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
        edges = np.uint8(np.absolute(edges))
    
    else:
        edges = gray
    
    return edges

def save_image(image, format='JPEG'):
    """Save image to bytes"""
    if image is None:
        return None
    
    # Convert to PIL if it's a numpy array
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    
    img_byte_arr = io.BytesIO()
    
    if format == 'JPEG':
        image.save(img_byte_arr, format='JPEG', quality=95)
    elif format == 'PNG':
        image.save(img_byte_arr, format='PNG')
    elif format == 'BMP':
        image.save(img_byte_arr, format='BMP')
    
    img_byte_arr.seek(0)
    return img_byte_arr

# Main application
def main():
    st.title("🖼️ Image Processing Toolkit")
    st.markdown("---")
    
    # Sidebar for operations
    st.sidebar.title("Operations")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "tiff"])
    
    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.session_state.original_image = image
        st.session_state.processed_image = image
        st.session_state.image_uploaded = True
        update_image_info()
    
    # Operation categories
    operation_category = st.sidebar.selectbox(
        "Select Operation Category",
        ["Image Info", "Color Conversions", "Transformations", "Filtering & Morphology", 
         "Enhancement", "Edge Detection", "Compression"]
    )
    
    # Initialize parameters
    params = {}
    
    # Operation-specific parameters
    if operation_category == "Image Info":
        if st.session_state.image_uploaded:
            st.sidebar.info("Image information will be displayed in the status bar.")
        else:
            st.sidebar.warning("Please upload an image first.")
    
    elif operation_category == "Color Conversions":
        color_conv = st.sidebar.selectbox(
            "Select Color Conversion",
            ["RGB to BGR", "BGR to RGB", "RGB to HSV", "HSV to RGB", 
             "RGB to YCbCr", "YCbCr to RGB", "RGB to Grayscale"]
        )
        
        if st.sidebar.button("Apply Color Conversion"):
            if st.session_state.image_uploaded:
                st.session_state.processed_image = convert_color(
                    st.session_state.original_image, color_conv
                )
                update_image_info()
    
    elif operation_category == "Transformations":
        transformation = st.sidebar.selectbox(
            "Select Transformation",
            ["Rotation", "Scaling", "Translation", "Affine Transform", "Perspective Transform"]
        )
        
        if transformation == "Rotation":
            angle = st.sidebar.slider("Rotation Angle", -180, 180, 0)
            params['angle'] = angle
        
        elif transformation == "Scaling":
            scale_x = st.sidebar.slider("Scale X", 0.1, 3.0, 1.0, 0.1)
            scale_y = st.sidebar.slider("Scale Y", 0.1, 3.0, 1.0, 0.1)
            params['scale_x'] = scale_x
            params['scale_y'] = scale_y
        
        elif transformation == "Translation":
            tx = st.sidebar.slider("Translate X", -100, 100, 0)
            ty = st.sidebar.slider("Translate Y", -100, 100, 0)
            params['tx'] = tx
            params['ty'] = ty
        
        if st.sidebar.button("Apply Transformation"):
            if st.session_state.image_uploaded:
                st.session_state.processed_image = apply_transformation(
                    st.session_state.original_image, transformation, **params
                )
                update_image_info()
    
    elif operation_category == "Filtering & Morphology":
        operation_type = st.sidebar.selectbox(
            "Select Operation Type",
            ["Filtering", "Morphology"]
        )
        
        if operation_type == "Filtering":
            filter_type = st.sidebar.selectbox(
                "Select Filter",
                ["Gaussian Blur", "Mean Filter", "Median Filter", "Sobel", "Laplacian"]
            )
            kernel_size = st.sidebar.slider("Kernel Size", 3, 15, 5, 2)
            params['kernel_size'] = kernel_size
            
            if st.sidebar.button("Apply Filter"):
                if st.session_state.image_uploaded:
                    st.session_state.processed_image = apply_filter(
                        st.session_state.original_image, filter_type, **params
                    )
                    update_image_info()
        
        else:
            morph_op = st.sidebar.selectbox(
                "Select Morphological Operation",
                ["Dilation", "Erosion", "Opening", "Closing"]
            )
            kernel_size = st.sidebar.slider("Kernel Size", 3, 15, 5, 2)
            params['kernel_size'] = kernel_size
            
            if st.sidebar.button("Apply Morphological Operation"):
                if st.session_state.image_uploaded:
                    st.session_state.processed_image = apply_morphology(
                        st.session_state.original_image, morph_op, **params
                    )
                    update_image_info()
    
    elif operation_category == "Enhancement":
        enhancement = st.sidebar.selectbox(
            "Select Enhancement",
            ["Histogram Equalization", "Contrast Stretching", "Sharpening"]
        )
        
        if st.sidebar.button("Apply Enhancement"):
            if st.session_state.image_uploaded:
                st.session_state.processed_image = apply_enhancement(
                    st.session_state.original_image, enhancement
                )
                update_image_info()
    
    elif operation_category == "Edge Detection":
        edge_method = st.sidebar.selectbox(
            "Select Edge Detection Method",
            ["Sobel", "Canny", "Laplacian"]
        )
        
        if edge_method == "Canny":
            threshold1 = st.sidebar.slider("Threshold 1", 0, 255, 100)
            threshold2 = st.sidebar.slider("Threshold 2", 0, 255, 200)
            params['threshold1'] = threshold1
            params['threshold2'] = threshold2
        
        if st.sidebar.button("Apply Edge Detection"):
            if st.session_state.image_uploaded:
                st.session_state.processed_image = apply_edge_detection(
                    st.session_state.original_image, edge_method, **params
                )
                update_image_info()
    
    elif operation_category == "Compression":
        format_type = st.sidebar.selectbox(
            "Select Format",
            ["JPEG", "PNG", "BMP"]
        )
        
        if st.sidebar.button("Apply Compression"):
            if st.session_state.image_uploaded:
                # For compression, we'll just convert the format
                # In a real application, you would adjust quality parameters
                st.session_state.processed_image = st.session_state.original_image
                update_image_info()
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        if st.session_state.original_image is not None:
            # Convert BGR to RGB for display
            original_rgb = cv2.cvtColor(st.session_state.original_image, cv2.COLOR_BGR2RGB)
            st.image(original_rgb, use_column_width=True, caption="Original Image")
        else:
            st.info("Please upload an image to get started")
    
    with col2:
        st.subheader("Processed Image")
        if st.session_state.processed_image is not None:
            # Handle different image types for display
            if isinstance(st.session_state.processed_image, np.ndarray):
                if len(st.session_state.processed_image.shape) == 3:
                    processed_rgb = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
                    st.image(processed_rgb, use_column_width=True, caption="Processed Image")
                else:
                    st.image(st.session_state.processed_image, use_column_width=True, 
                            caption="Processed Image", clamp=True)
            else:
                st.image(st.session_state.processed_image, use_column_width=True, caption="Processed Image")
        else:
            st.info("Processed image will appear here")
    
    # Status bar
    st.markdown("---")
    st.subheader("Image Information")
    
    if st.session_state.image_info:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Dimensions:**")
            st.write(f"Width: {st.session_state.image_info.get('width', 'N/A')} px")
            st.write(f"Height: {st.session_state.image_info.get('height', 'N/A')} px")
            st.write(f"Channels: {st.session_state.image_info.get('channels', 'N/A')}")
        
        with col2:
            st.write("**Format Info:**")
            st.write(f"Mode: {st.session_state.image_info.get('mode', 'N/A')}")
            st.write(f"DPI: {st.session_state.image_info.get('dpi', 'N/A')}")
        
        with col3:
            st.write("**File Sizes:**")
            st.write(f"JPEG: {st.session_state.image_info.get('jpg_size', 0):.2f} KB")
            st.write(f"PNG: {st.session_state.image_info.get('png_size', 0):.2f} KB")
            st.write(f"BMP: {st.session_state.image_info.get('bmp_size', 0):.2f} KB")
    
    # Save options
    st.markdown("---")
    st.subheader("Save Options")
    
    if st.session_state.processed_image is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            jpg_bytes = save_image(st.session_state.processed_image, 'JPEG')
            if jpg_bytes:
                st.download_button(
                    label="Download as JPEG",
                    data=jpg_bytes,
                    file_name="processed_image.jpg",
                    mime="image/jpeg"
                )
        
        with col2:
            png_bytes = save_image(st.session_state.processed_image, 'PNG')
            if png_bytes:
                st.download_button(
                    label="Download as PNG",
                    data=png_bytes,
                    file_name="processed_image.png",
                    mime="image/png"
                )
        
        with col3:
            bmp_bytes = save_image(st.session_state.processed_image, 'BMP')
            if bmp_bytes:
                st.download_button(
                    label="Download as BMP",
                    data=bmp_bytes,
                    file_name="processed_image.bmp",
                    mime="image/bmp"
                )
    else:
        st.info("Process an image to enable download options")

if __name__ == "__main__":
    main()
