import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from utils import border_mode, rgb_to_gray_manual, rgb_to_hsv_manual, rgb_to_ycbcr_manual

# Initialize session state
if "processed" not in st.session_state:
    st.session_state.processed = None

# ---------------------------- Page Config ----------------------------
st.set_page_config(page_title="Image Processing Studio", layout="wide")
st.title("🖼️ Image Processing Studio")

# ---------------------------- Menu Bar ----------------------------
col1, col2, col3 = st.columns(3)
with col1:
    uploaded_file = st.file_uploader("📂 Open Image", type=["jpg", "png", "bmp"])
with col2:
    save_btn = st.button("💾 Save Processed Image")
with col3:
    if st.button("❌ Exit App"):
        st.stop()

# ---------------------------- Load Image ----------------------------
original_img = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    original_img = np.array(image)
    st.session_state["original"] = original_img

# ---------------------------- Sidebar Menu ----------------------------
st.sidebar.header("⚙️ Operations Menu")
operation_category = st.sidebar.radio("Select Operation", 
    ["Image Info", "Color Conversions", "Transformations", 
     "Filtering & Morphology", "Enhancement", 
     "Edge Detection", "Compression"])

processed_img = st.session_state.processed

# ---------------------------- Image Info ----------------------------
if operation_category == "Image Info" and original_img is not None:
    st.subheader("📊 Image Information")
    h, w = original_img.shape[:2]
    c = 1 if len(original_img.shape) == 2 else original_img.shape[2]
    file_size_kb = len(uploaded_file.getbuffer()) // 1024
    dpi = image.info.get("dpi", ("N/A", "N/A"))
    st.write(f"**Dimensions:** {h} x {w} x {c}")
    st.write(f"**File Size:** {file_size_kb} KB")
    st.write(f"**Format:** {uploaded_file.type}")
    st.write(f"**DPI:** {dpi}")
    st.write(f"**Color Channels:** {c}")

# ---------------------------- Color Conversions ----------------------------
if operation_category == "Color Conversions" and original_img is not None:
    mode = st.sidebar.radio("Implementation", ["OpenCV", "Manual"])
    choice = st.sidebar.selectbox("Choose Conversion", 
        ["RGB → BGR", "RGB → HSV", "RGB → YCbCr", "RGB → Grayscale"])
    
    if mode == "OpenCV":
        if choice == "RGB → BGR":
            processed_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        elif choice == "RGB → HSV":
            processed_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2HSV)
        elif choice == "RGB → YCbCr":
            processed_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2YCrCb)
        elif choice == "RGB → Grayscale":
            processed_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    else:
        if choice == "RGB → Grayscale":
            processed_img = rgb_to_gray_manual(original_img)
        elif choice == "RGB → HSV":
            processed_img = rgb_to_hsv_manual(original_img)
        elif choice == "RGB → YCbCr":
            processed_img = rgb_to_ycbcr_manual(original_img)
        elif choice == "RGB → BGR":
            processed_img = original_img[..., ::-1]

# ---------------------------- Transformations ----------------------------
if operation_category == "Transformations" and original_img is not None:
    transform = st.sidebar.selectbox("Choose Transform", 
        ["Translation","Rotation","Scaling (Resize)","Affine Transform", "Perspective Transform"])
    
    h, w = original_img.shape[:2]
    if transform == "Translation":
        dx = st.sidebar.slider("Shift X", -w//2, w//2, 0)
        dy = st.sidebar.slider("Shift Y", -h//2, h//2, 0)
        border_choice = st.sidebar.selectbox("Border Mode", ["Constant (black)", "Reflect", "Replicate", "Wrap"])
        M = np.float32([[1,0,dx],[0,1,dy]])
        warp_kwargs = {"borderMode": border_mode(border_choice)}
        if border_choice == "Constant (black)":
            warp_kwargs["borderValue"] = 0
        processed_img = cv2.warpAffine(original_img, M, (w,h), **warp_kwargs)
    
    elif transform == "Rotation":
        angle = st.sidebar.slider("Angle (°)", -180,180,0)
        scale = st.sidebar.slider("Scale",0.1,3.0,1.0)
        cx = st.sidebar.slider("Center X", 0,w,w//2)
        cy = st.sidebar.slider("Center Y", 0,h,h//2)
        border_choice = st.sidebar.selectbox("Border Mode", ["Constant (black)", "Reflect", "Replicate", "Wrap"], key="rot_border")
        M = cv2.getRotationMatrix2D((cx,cy), angle, scale)
        warp_kwargs = {"borderMode": border_mode(border_choice)}
        if border_choice == "Constant (black)":
            warp_kwargs["borderValue"] = 0
        processed_img = cv2.warpAffine(original_img, M, (w,h), **warp_kwargs)
    
    elif transform == "Scaling (Resize)":
        fx = st.sidebar.slider("Scale X", 0.1,3.0,1.0)
        fy = st.sidebar.slider("Scale Y", 0.1,3.0,1.0)
        processed_img = cv2.resize(original_img, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    
    elif transform == "Affine Transform":
        src = np.float32([[0,0],[w-1,0],[0,h-1]])
        dst = np.float32([[0,0],[w-1,0],[int(0.3*w),h-1]])
        M = cv2.getAffineTransform(src,dst)
        processed_img = cv2.warpAffine(original_img,M,(w,h))
    
    elif transform == "Perspective Transform":
        src = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
        dst = np.float32([[0,0],[w-1,0],[int(0.8*w),h-1],[int(0.2*w),h-1]])
        M = cv2.getPerspectiveTransform(src,dst)
        processed_img = cv2.warpPerspective(original_img,M,(w,h))

# ---------------------------- Filtering & Morphology ----------------------------
if operation_category == "Filtering & Morphology" and original_img is not None:
    choice = st.sidebar.radio("Category", ["Filtering", "Morphology"])
    gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    
    if choice == "Filtering":
        filter_type = st.sidebar.selectbox("Filter", ["Gaussian Blur","Mean Blur","Median Blur","Sobel","Laplacian"])
        if filter_type == "Gaussian Blur":
            k = st.sidebar.slider("Kernel",1,31,5,2)
            processed_img = cv2.GaussianBlur(original_img,(k,k),0)
        elif filter_type == "Mean Blur":
            k = st.sidebar.slider("Kernel",1,31,5,2)
            processed_img = cv2.blur(original_img,(k,k))
        elif filter_type == "Median Blur":
            k = st.sidebar.slider("Kernel",1,31,5,2)
            processed_img = cv2.medianBlur(original_img,k)
        elif filter_type == "Sobel":
            dx = st.sidebar.slider("dx",0,2,1)
            dy = st.sidebar.slider("dy",0,2,1)
            k = st.sidebar.selectbox("Kernel",[1,3,5,7],key="sobel_ksize")
            processed_img = cv2.Sobel(gray,cv2.CV_64F,dx,dy,ksize=k)
            processed_img = cv2.convertScaleAbs(processed_img)
        elif filter_type == "Laplacian":
            k = st.sidebar.selectbox("Kernel",[1,3,5,7],key="lap_ksize")
            processed_img = cv2.Laplacian(gray,cv2.CV_64F,ksize=k)
            processed_img = cv2.convertScaleAbs(processed_img)
    else: # Morphology
        morph_type = st.sidebar.selectbox("Morph", ["Dilation","Erosion","Opening","Closing"])
        k = st.sidebar.slider("Kernel Size",1,21,3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(k,k))
        if morph_type=="Dilation":
            processed_img = cv2.dilate(gray,kernel)
        elif morph_type=="Erosion":
            processed_img = cv2.erode(gray,kernel)
        elif morph_type=="Opening":
            processed_img = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        elif morph_type=="Closing":
            processed_img = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

# ---------------------------- Enhancement ----------------------------
if operation_category == "Enhancement" and original_img is not None:
    enhance_type = st.sidebar.selectbox("Enhancement", ["Histogram Equalization","Contrast Stretching","Sharpening"])
    if enhance_type=="Histogram Equalization":
        gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
        processed_img = cv2.equalizeHist(gray)
    elif enhance_type=="Contrast Stretching":
        min_val,max_val = np.min(original_img), np.max(original_img)
        processed_img = ((original_img - min_val)/(max_val-min_val)*255).astype(np.uint8)
    elif enhance_type=="Sharpening":
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        processed_img = cv2.filter2D(original_img,-1,kernel)

# ---------------------------- Edge Detection ----------------------------
if operation_category == "Edge Detection" and original_img is not None:
    edge_type = st.sidebar.selectbox("Edge Detection", ["Sobel","Canny","Laplacian"])
    gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    if edge_type=="Sobel":
        dx = st.sidebar.slider("dx",0,2,1)
        dy = st.sidebar.slider("dy",0,2,1)
        k = st.sidebar.selectbox("Kernel",[1,3,5,7],key="edge_sobel")
        processed_img = cv2.Sobel(gray,cv2.CV_64F,dx,dy,ksize=k)
        processed_img = cv2.convertScaleAbs(processed_img)
    elif edge_type=="Canny":
        t1 = st.sidebar.slider("Threshold 1",0,500,100)
        t2 = st.sidebar.slider("Threshold 2",0,500,200)
        processed_img = cv2.Canny(gray,t1,t2)
    elif edge_type=="Laplacian":
        k = st.sidebar.selectbox("Kernel",[1,3,5,7],key="edge_lap")
        processed_img = cv2.Laplacian(gray,cv2.CV_64F,ksize=k)
        processed_img = cv2.convertScaleAbs(processed_img)

# ---------------------------- Compression ----------------------------
if operation_category == "Compression" and (original_img is not None or processed_img is not None):
    st.sidebar.subheader("Compression Options")
    img_to_save = processed_img if processed_img is not None else original_img
    fmt = st.sidebar.selectbox("Format", ["JPG","PNG","BMP"], key="comp_format")
    quality = 95
    if fmt=="JPG":
        quality = st.sidebar.slider("JPEG Quality",10,100,95,key="comp_quality")
    if st.sidebar.button("Save Image", key="comp_save"):
        if len(img_to_save.shape)==2:
            save_img = Image.fromarray(img_to_save)
        else:
            save_img = Image.fromarray(cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
        buffer = io.BytesIO()
        if fmt=="JPG":
            save_img.save(buffer, format="JPEG", quality=quality)
        else:
            save_img.save(buffer, format=fmt)
        st.success(f"Saved {fmt} | Size: {buffer.getbuffer().nbytes//1024} KB")
        st.download_button("📥 Download", buffer, file_name=f"image.{fmt.lower()}", mime=f"image/{fmt.lower()}", key="comp_dl")

# ---------------------------- Display ----------------------------
st.session_state.processed = processed_img
col1, col2 = st.columns(2)
with col1:
    if original_img is not None:
        st.image(original_img, caption="Original Image", use_column_width=True)
with col2:
    img_show = processed_img if processed_img is not None else original_img
    if img_show is not None:
        st.image(img_show, caption="Processed / Final Image", use_column_width=True)
    else:
        st.info("Processed image will appear here after applying operations.")

# ---------------------------- Status Bar ----------------------------
st.markdown("---")
if original_img is not None:
    h,w = original_img.shape[:2]
    c = 1 if len(original_img.shape)==2 else original_img.shape[2]
    file_size_kb = len(uploaded_file.getbuffer())//1024
    st.markdown(f"📏 Dimensions: {h}x{w}x{c} | 🗂 Size: {file_size_kb} KB | Format: {uploaded_file.type}")
else:
    st.markdown("ℹ️ Upload an image to see details here.")
