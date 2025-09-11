import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import time
from matplotlib import pyplot as plt

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
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .image-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    .image-box {
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 10px;
        text-align: center;
        margin: 0 10px;
    }
    .status-bar {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
        font-family: monospace;
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
if 'image_info' not in st.session_state:
    st.session_state.image_info = {}

def normalize_image_for_display(image):
    """Normalize image to 0-255 range for display"""
    if image is None:
        return None
    
    # Handle different data types
    if image.dtype != np.uint8:
        # Normalize to 0-255 range
        if np.max(image) > 1 or np.min(image) < 0:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = image.astype(np.uint8)
    
    return image

def get_image_info(image, filename):
    """Extract image information"""
    if image is None:
        return {}
    
    if len(image.shape) == 3:
        height, width, channels = image.shape
    else:
        height, width = image.shape
        channels = 1
    
    # Create a temporary file to get file size
    if st.session_state.original_image is not None:
        pil_image = Image.fromarray(cv2.cvtColor(st.session_state.original_image, cv2.COLOR_BGR2RGB))
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        file_size = len(img_byte_arr.getvalue())
    else:
        file_size = 0
    
    return {
        'filename': filename,
        'dimensions': f"{width} x {height}",
        'channels': channels,
        'file_size': f"{file_size / 1024:.2f} KB" if file_size > 0 else "N/A",
        'color_mode': 'Color' if channels == 3 else 'Grayscale'
    }

def load_image(uploaded_file):
    """Load image from uploaded file"""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.session_state.original_image = image
    st.session_state.processed_image = image.copy()
    st.session_state.image_info = get_image_info(image, uploaded_file.name)
    return image

def save_image(image, filename, format):
    """Save processed image"""
    if image is not None:
        # Normalize image first
        image = normalize_image_for_display(image)
        
        # Convert BGR to RGB for PIL
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        pil_image = Image.fromarray(image_rgb)
        buf = io.BytesIO()
        
        if format == 'JPEG':
            pil_image.save(buf, format='JPEG', quality=95)
        elif format == 'PNG':
            pil_image.save(buf, format='PNG')
        elif format == 'BMP':
            pil_image.save(buf, format='BMP')
        
        byte_im = buf.getvalue()
        
        st.download_button(
            label=f"Download as {format}",
            data=byte_im,
            file_name=f"processed_{filename.split('.')[0]}.{format.lower()}",
            mime=f"image/{format.lower()}"
        )

def apply_color_conversion(image, conversion):
    """Apply color space conversion"""
    if image is None:
        return None
    
    conversions = {
        'RGB ↔ BGR': cv2.COLOR_RGB2BGR,
        'RGB ↔ HSV': cv2.COLOR_RGB2HSV,
        'RGB ↔ YCbCr': cv2.COLOR_RGB2YCrCb,
        'RGB ↔ Grayscale': cv2.COLOR_RGB2GRAY
    }
    
    if conversion in conversions:
        # Convert to RGB first if the image is grayscale
        if len(image.shape) == 2 and conversion != 'RGB ↔ Grayscale':
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        if conversion == 'RGB ↔ Grayscale':
            result = cv2.cvtColor(image, conversions[conversion])
        else:
            # Convert to RGB first
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            result = cv2.cvtColor(image_rgb, conversions[conversion])
        
        return normalize_image_for_display(result)
    
    return image

def apply_transformation(image, transformation, params):
    """Apply geometric transformation"""
    if image is None:
        return None
    
    height, width = image.shape[:2]
    
    if transformation == 'Rotation':
        angle = params.get('angle', 0)
        scale = params.get('scale', 1.0)
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        result = cv2.warpAffine(image, matrix, (width, height))
    
    elif transformation == 'Scaling':
        scale_x = params.get('scale_x', 1.0)
        scale_y = params.get('scale_y', 1.0)
        new_width = int(width * scale_x)
        new_height = int(height * scale_y)
        result = cv2.resize(image, (new_width, new_height))
    
    elif transformation == 'Translation':
        tx = params.get('tx', 0)
        ty = params.get('ty', 0)
        matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        result = cv2.warpAffine(image, matrix, (width, height))
    
    elif transformation == 'Affine Transform':
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
        matrix = cv2.getAffineTransform(pts1, pts2)
        result = cv2.warpAffine(image, matrix, (width, height))
    
    elif transformation == 'Perspective Transform':
        pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
        pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(image, matrix, (300, 300))
    
    else:
        result = image
    
    return normalize_image_for_display(result)

def apply_filter(image, filter_type, params):
    """Apply image filter"""
    if image is None:
        return None
    
    kernel_size = params.get('kernel_size', 3)
    
    if filter_type == 'Gaussian':
        result = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    elif filter_type == 'Mean':
        result = cv2.blur(image, (kernel_size, kernel_size))
    
    elif filter_type == 'Median':
        result = cv2.medianBlur(image, kernel_size)
    
    elif filter_type == 'Sobel':
        dx = params.get('dx', 1)
        dy = params.get('dy', 0)
        sobel = cv2.Sobel(image, cv2.CV_64F, dx, dy, ksize=kernel_size)
        result = np.absolute(sobel)
    
    elif filter_type == 'Laplacian':
        laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=kernel_size)
        result = np.absolute(laplacian)
    
    else:
        result = image
    
    return normalize_image_for_display(result)

def apply_morphology(image, operation, params):
    """Apply morphological operation"""
    if image is None:
        return None
    
    kernel_size = params.get('kernel_size', 3)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    if operation == 'Dilation':
        result = cv2.dilate(image, kernel, iterations=1)
    
    elif operation == 'Erosion':
        result = cv2.erode(image, kernel, iterations=1)
    
    elif operation == 'Opening':
        result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    elif operation == 'Closing':
        result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    else:
        result = image
    
    return normalize_image_for_display(result)

def apply_enhancement(image, enhancement, params):
    """Apply image enhancement"""
    if image is None:
        return None
    
    if enhancement == 'Histogram Equalization':
        if len(image.shape) == 2:
            result = cv2.equalizeHist(image)
        else:
            # For color images, convert to YUV and equalize the Y channel
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    elif enhancement == 'Contrast Stretching':
        # Simple contrast stretching
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:  # Avoid division by zero
            stretched = (image - min_val) * (255.0 / (max_val - min_val))
            result = stretched.astype(np.uint8)
        else:
            result = image
    
    elif enhancement == 'Sharpening':
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        result = cv2.filter2D(image, -1, kernel)
    
    else:
        result = image
    
    return normalize_image_for_display(result)

def apply_edge_detection(image, method, params):
    """Apply edge detection"""
    if image is None:
        return None
    
    if method == 'Sobel':
        dx = params.get('dx', 1)
        dy = params.get('dy', 1)
        sobelx = cv2.Sobel(image, cv2.CV_64F, dx, 0)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, dy)
        result = cv2.magnitude(sobelx, sobely)
    
    elif method == 'Canny':
        threshold1 = params.get('threshold1', 100)
        threshold2 = params.get('threshold2', 200)
        result = cv2.Canny(image, threshold1, threshold2)
    
    elif method == 'Laplacian':
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        result = np.absolute(laplacian)
    
    else:
        result = image
    
    return normalize_image_for_display(result)

# Main application
def main():
    st.markdown('<h1 class="main-header">Image Processing Toolkit</h1>', unsafe_allow_html=True)
    
    # Sidebar for operations
    with st.sidebar:
        st.header("Operations")
        
        # File operations
        st.subheader("File")
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png', 'bmp'])
        
        if uploaded_file is not None:
            if st.session_state.original_image is None:
                load_image(uploaded_file)
            
            if st.button("Reset Image"):
                st.session_state.processed_image = st.session_state.original_image.copy()
                st.rerun()
        
        # Image Info
        st.subheader("Image Info")
        if st.session_state.original_image is not None:
            st.write(f"Filename: {st.session_state.image_info.get('filename', 'N/A')}")
            st.write(f"Dimensions: {st.session_state.image_info.get('dimensions', 'N/A')}")
            st.write(f"Channels: {st.session_state.image_info.get('channels', 'N/A')}")
            st.write(f"Color Mode: {st.session_state.image_info.get('color_mode', 'N/A')}")
            st.write(f"File Size: {st.session_state.image_info.get('file_size', 'N/A')}")
        
        # Color Conversions
        st.subheader("Color Conversions")
        color_conv = st.selectbox(
            "Select Color Conversion",
            ["None", "RGB ↔ BGR", "RGB ↔ HSV", "RGB ↔ YCbCr", "RGB ↔ Grayscale"]
        )
        
        # Transformations
        st.subheader("Transformations")
        transformation = st.selectbox(
            "Select Transformation",
            ["None", "Rotation", "Scaling", "Translation", "Affine Transform", "Perspective Transform"]
        )
        
        # Parameters for transformations
        transformation_params = {}
        if transformation == "Rotation":
            transformation_params['angle'] = st.slider("Rotation Angle", -180, 180, 0)
            transformation_params['scale'] = st.slider("Scale", 0.5, 2.0, 1.0)
        elif transformation == "Scaling":
            transformation_params['scale_x'] = st.slider("Scale X", 0.1, 3.0, 1.0)
            transformation_params['scale_y'] = st.slider("Scale Y", 0.1, 3.0, 1.0)
        elif transformation == "Translation":
            transformation_params['tx'] = st.slider("Translate X", -100, 100, 0)
            transformation_params['ty'] = st.slider("Translate Y", -100, 100, 0)
        
        # Filtering & Morphology
        st.subheader("Filtering & Morphology")
        filter_type = st.selectbox(
            "Select Filter",
            ["None", "Gaussian", "Mean", "Median", "Sobel", "Laplacian"]
        )
        
        morphology = st.selectbox(
            "Morphological Operation",
            ["None", "Dilation", "Erosion", "Opening", "Closing"]
        )
        
        # Kernel size for filters and morphology
        kernel_size = st.slider("Kernel Size", 1, 15, 3, step=2)
        
        filter_params = {'kernel_size': kernel_size}
        morphology_params = {'kernel_size': kernel_size}
        
        # Enhancement
        st.subheader("Enhancement")
        enhancement = st.selectbox(
            "Select Enhancement",
            ["None", "Histogram Equalization", "Contrast Stretching", "Sharpening"]
        )
        
        # Edge Detection
        st.subheader("Edge Detection")
        edge_detection = st.selectbox(
            "Select Edge Detection",
            ["None", "Sobel", "Canny", "Laplacian"]
        )
        
        # Parameters for edge detection
        edge_params = {}
        if edge_detection == "Canny":
            edge_params['threshold1'] = st.slider("Threshold 1", 0, 255, 100)
            edge_params['threshold2'] = st.slider("Threshold 2", 0, 255, 200)
        elif edge_detection == "Sobel":
            edge_params['dx'] = st.slider("dx", 0, 1, 1)
            edge_params['dy'] = st.slider("dy", 0, 1, 1)
        
        # Compression
        st.subheader("Compression")
        compression_format = st.selectbox(
            "Save Format",
            ["None", "JPEG", "PNG", "BMP"]
        )
        
        # Apply operations button
        apply_operations = st.button("Apply Operations")
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        if st.session_state.original_image is not None:
            st.image(cv2.cvtColor(st.session_state.original_image, cv2.COLOR_BGR2RGB), 
                    use_column_width=True)
    
    with col2:
        st.subheader("Processed Image")
        if st.session_state.processed_image is not None:
            # Convert to RGB for display
            if len(st.session_state.processed_image.shape) == 2:
                display_image = st.session_state.processed_image
            else:
                display_image = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
            
            st.image(display_image, use_column_width=True)
    
    # Apply operations when button is clicked
    if apply_operations and st.session_state.original_image is not None:
        processed = st.session_state.original_image.copy()
        
        # Apply color conversion
        if color_conv != "None":
            processed = apply_color_conversion(processed, color_conv)
        
        # Apply transformation
        if transformation != "None":
            processed = apply_transformation(processed, transformation, transformation_params)
        
        # Apply filter
        if filter_type != "None":
            processed = apply_filter(processed, filter_type, filter_params)
        
        # Apply morphological operation
        if morphology != "None":
            processed = apply_morphology(processed, morphology, morphology_params)
        
        # Apply enhancement
        if enhancement != "None":
            processed = apply_enhancement(processed, enhancement, {})
        
        # Apply edge detection
        if edge_detection != "None":
            processed = apply_edge_detection(processed, edge_detection, edge_params)
        
        # Update processed image
        st.session_state.processed_image = processed
        
        # Refresh the page to show updated image
        st.rerun()
    
    # Save processed image
    if compression_format != "None" and st.session_state.processed_image is not None:
        st.subheader("Save Processed Image")
        save_image(st.session_state.processed_image, 
                  st.session_state.image_info.get('filename', 'image'), 
                  compression_format)
    
    # Status bar
    st.markdown("---")
    st.markdown('<div class="status-bar">', unsafe_allow_html=True)
    st.write("Status: Ready" if st.session_state.original_image is not None else "Status: No image loaded")
    
    if st.session_state.original_image is not None:
        st.write(f"Original Dimensions: {st.session_state.image_info.get('dimensions', 'N/A')}")
        
        if st.session_state.processed_image is not None:
            if len(st.session_state.processed_image.shape) == 3:
                h, w, c = st.session_state.processed_image.shape
            else:
                h, w = st.session_state.processed_image.shape
                c = 1
            st.write(f"Processed Dimensions: {w} x {h} x {c}")
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()