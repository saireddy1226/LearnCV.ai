
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional

# ------------------------------
# Image Info
# ------------------------------
def get_image_info(img: np.ndarray, source_bytes: Optional[bytes] = None, file_ext: Optional[str] = None) -> Dict[str, Any]:
    if img is None:
        return {}
    h, w = img.shape[:2]
    channels = 1 if len(img.shape) == 2 else img.shape[2]
    file_size = len(source_bytes) if source_bytes is not None else None
    fmt = file_ext.lower().strip('.') if file_ext else None
    info = {
        "height": h,
        "width": w,
        "channels": channels,
        "dimensions": (h, w, channels),
        "file_format": fmt,
        "file_size_bytes": file_size,
        "file_size_kb": round(file_size/1024, 2) if file_size is not None else None,
        "dpi_ppi": None  # OpenCV does not store DPI; usually part of metadata
    }
    return info

# ------------------------------
# Color Conversions
# ------------------------------
def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def rgb_to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def rgb_to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

def hsv_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

def rgb_to_ycrcb(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

def ycrcb_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)

def rgb_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gray_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

# ------------------------------
# Transforms
# ------------------------------
def rotate_image(img: np.ndarray, angle: float, center: Optional[Tuple[int, int]] = None, scale: float = 1.0) -> np.ndarray:
    (h, w) = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

def scale_image(img: np.ndarray, fx: float, fy: float) -> np.ndarray:
    return cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR if fx >= 1 or fy >= 1 else cv2.INTER_AREA)

def translate_image(img: np.ndarray, tx: float, ty: float) -> np.ndarray:
    (h, w) = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted = cv2.warpAffine(img, M, (w, h))
    return shifted

def affine_transform(img: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    M = cv2.getAffineTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))
    (h, w) = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h))

def perspective_transform(img: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    M = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))
    (h, w) = img.shape[:2]
    return cv2.warpPerspective(img, M, (w, h))

# ------------------------------
# Bitwise Ops
# ------------------------------
def bitwise_and(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(a, b)

def bitwise_or(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cv2.bitwise_or(a, b)

def bitwise_xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cv2.bitwise_xor(a, b)

def bitwise_not(a: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(a)

# ------------------------------
# Filtering
# ------------------------------
def mean_filter(img: np.ndarray, k: int) -> np.ndarray:
    return cv2.blur(img, (k, k))

def gaussian_filter(img: np.ndarray, k: int, sigma: float = 0) -> np.ndarray:
    k = k + 1 if k % 2 == 0 else k  # ensure odd
    return cv2.GaussianBlur(img, (k, k), sigma)

def median_filter(img: np.ndarray, k: int) -> np.ndarray:
    k = k + 1 if k % 2 == 0 else k  # ensure odd
    return cv2.medianBlur(img, k)

# ------------------------------
# Edges
# ------------------------------
def sobel_edges(img_gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mag = np.uint8(255 * mag / (mag.max() + 1e-6))
    return mag

def laplacian_edges(img_gray: np.ndarray) -> np.ndarray:
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    lap = np.uint8(255 * (np.abs(lap) / (np.max(np.abs(lap)) + 1e-6)))
    return lap

def canny_edges(img_gray: np.ndarray, t1: int, t2: int) -> np.ndarray:
    return cv2.Canny(img_gray, t1, t2)

# ------------------------------
# Morphology
# ------------------------------
def morphology(img: np.ndarray, op: str, k: int, iterations: int = 1) -> np.ndarray:
    kernel = np.ones((k, k), np.uint8)
    if op == "dilate":
        return cv2.dilate(img, kernel, iterations=iterations)
    elif op == "erode":
        return cv2.erode(img, kernel, iterations=iterations)
    elif op == "open":
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif op == "close":
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    else:
        return img

# ------------------------------
# Enhancement
# ------------------------------
def histogram_equalization(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return cv2.equalizeHist(img)
    else:
        # Convert to YCrCb and equalize Y channel
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

def contrast_stretch(img: np.ndarray, low_perc=2, high_perc=98) -> np.ndarray:
    # percentile-based stretching
    if len(img.shape) == 2:
        lo, hi = np.percentile(img, (low_perc, high_perc))
        out = (img - lo) * (255.0/(hi - lo + 1e-6))
        return np.clip(out, 0, 255).astype(np.uint8)
    else:
        out = np.zeros_like(img)
        for c in range(3):
            lo, hi = np.percentile(img[:,:,c], (low_perc, high_perc))
            chan = (img[:,:,c] - lo) * (255.0/(hi - lo + 1e-6))
            out[:,:,c] = np.clip(chan, 0, 255)
        return out.astype(np.uint8)

def sharpen(img: np.ndarray, amount: float = 1.0) -> np.ndarray:
    # Unsharp masking
    blurred = cv2.GaussianBlur(img, (0,0), sigmaX=3)
    sharp = cv2.addWeighted(img, 1+amount, blurred, -amount, 0)
    return sharp

# ------------------------------
# Helpers
# ------------------------------
def ensure_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """Streamlit expects RGB for display."""
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def ensure_gray(img_rgb: np.ndarray) -> np.ndarray:
    if len(img_rgb.shape) == 2:
        return img_rgb
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

def to_3channel(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def split_screen_compare(left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
    # Assumes both are RGB with same H,W
    left = to_3channel(left_img)
    right = to_3channel(right_img)
    h, w = left.shape[:2]
    out = left.copy()
    out[:, w//2:] = right[:, w//2:]
    return out

def encode_format(img_rgb: np.ndarray, ext: str = ".png") -> bytes:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ret, buf = cv2.imencode(ext, bgr)
    return buf.tobytes() if ret else None
