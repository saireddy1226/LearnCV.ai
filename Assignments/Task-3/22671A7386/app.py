
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# ---------------------------
# CV Toolkit Class
# ---------------------------
class CVToolkit:
    def __init__(self):
        self.original = None
        self.processed = None

    def load_image(self, file):
        if file:
            pil_img = Image.open(file).convert('RGB')
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            self.original = img
            return img
        return None

    def prepare_display(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img

    # ---------- Color Conversions ----------
    def convert_to_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def convert_to_hsv(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def sepia_filter(self, img):
        kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        sepia = cv2.transform(img, kernel)
        return np.clip(sepia, 0, 255).astype(np.uint8)

    def invert_colors(self, img):
        return cv2.bitwise_not(img)

    # ---------- Transformations ----------
    def rotate(self, img, angle, expand=False):
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        if expand:
            cos = abs(M[0, 0])
            sin = abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            return cv2.warpAffine(img, M, (new_w, new_h), borderValue=(255, 255, 255))
        return cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

    def scale(self, img, factor):
        h, w = img.shape[:2]
        new_w, new_h = int(w * factor), int(h * factor)
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full_like(img, 255)
        y_off = max(0, (h - new_h) // 2)
        x_off = max(0, (w - new_w) // 2)

        if new_h <= h and new_w <= w:
            canvas[y_off:y_off + new_h, x_off:x_off + new_w] = scaled
        else:
            y_crop = max(0, (new_h - h) // 2)
            x_crop = max(0, (new_w - w) // 2)
            canvas = scaled[y_crop:y_crop + h, x_crop:x_crop + w]
        return canvas

    def translate(self, img, tx, ty):
        h, w = img.shape[:2]
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

    def flip(self, img, horizontal=False, vertical=False):
        if horizontal and vertical:
            return cv2.flip(img, -1)
        if horizontal:
            return cv2.flip(img, 1)
        if vertical:
            return cv2.flip(img, 0)
        return img

    # ---------- Filters ----------
    def gaussian_blur(self, img, ksize=5):
        if ksize % 2 == 0:
            ksize += 1
        return cv2.GaussianBlur(img, (ksize, ksize), 0)

    def sharpen(self, img, strength=1.0):
        kernel = np.array([
            [0, -strength, 0],
            [-strength, 1 + 4 * strength, -strength],
            [0, -strength, 0]
        ], np.float32)
        sharpened = cv2.filter2D(img, -1, kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def histogram_equalize(self, img):
        if len(img.shape) == 3:
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return cv2.equalizeHist(img)

    # ---------- Edge Detection ----------
    def sobel_edge(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sx**2 + sy**2)
        return np.clip(edges, 0, 255).astype(np.uint8)

    def canny_edge(self, img, low=50, high=150):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        return cv2.Canny(gray, low, high)

# ---------------------------
# Streamlit App UI
# ---------------------------
def main():
    st.set_page_config(page_title="CV Toolkit", layout="wide")
    st.title("🖌️ CV Toolkit - Image Processing")

    if 'toolkit' not in st.session_state:
        st.session_state.toolkit = CVToolkit()

    toolkit = st.session_state.toolkit

    file = st.sidebar.file_uploader("📤 Upload Image", type=['png', 'jpg', 'jpeg', 'bmp'])

    if file:
        img = toolkit.load_image(file)
        st.sidebar.success(f"Loaded: {file.name}")
        st.sidebar.info(f"Dimensions: {img.shape[1]}x{img.shape[0]}")

        # Color Conversion
        st.sidebar.subheader("🎨 Color Conversion")
        color_mode = st.sidebar.selectbox("Mode", ["None", "Grayscale", "HSV", "Sepia", "Invert"])
        if color_mode == "Grayscale":
            toolkit.processed = toolkit.convert_to_gray(img)
        elif color_mode == "HSV":
            toolkit.processed = toolkit.convert_to_hsv(img)
        elif color_mode == "Sepia":
            toolkit.processed = toolkit.sepia_filter(img)
        elif color_mode == "Invert":
            toolkit.processed = toolkit.invert_colors(img)

        # Transformations
        st.sidebar.subheader("🔄 Transformations")
        angle = st.sidebar.slider("Rotate", -180, 180, 0)
        if st.sidebar.button("Apply Rotation"):
            toolkit.processed = toolkit.rotate(img, angle, expand=True)

        scale_factor = st.sidebar.slider("Scale", 0.1, 3.0, 1.0)
        if st.sidebar.button("Apply Scaling"):
            toolkit.processed = toolkit.scale(img, scale_factor)

        tx = st.sidebar.slider("Translate X", -100, 100, 0)
        ty = st.sidebar.slider("Translate Y", -100, 100, 0)
        if st.sidebar.button("Apply Translation"):
            toolkit.processed = toolkit.translate(img, tx, ty)

        # Filters
        st.sidebar.subheader("🧪 Filters")
        blur_ksize = st.sidebar.slider("Gaussian Blur (Kernel)", 1, 31, 5, step=2)
        if st.sidebar.button("Apply Blur"):
            toolkit.processed = toolkit.gaussian_blur(img, blur_ksize)

        sharpen_strength = st.sidebar.slider("Sharpen Strength", 0.1, 3.0, 1.0)
        if st.sidebar.button("Apply Sharpen"):
            toolkit.processed = toolkit.sharpen(img, sharpen_strength)

        # Edge Detection
        st.sidebar.subheader("📐 Edge Detection")
        low = st.sidebar.slider("Canny Low Threshold", 0, 255, 50)
        high = st.sidebar.slider("Canny High Threshold", 0, 255, 150)
        if st.sidebar.button("Apply Canny"):
            toolkit.processed = toolkit.canny_edge(img, low, high)

        if st.sidebar.button("Apply Sobel"):
            toolkit.processed = toolkit.sobel_edge(img)

        # Display images
        col1, col2 = st.columns(2)
        with col1:
            st.image(toolkit.prepare_display(toolkit.original), caption="Original", use_column_width=True
```
