# utils.py
import cv2
import numpy as np

# ----------------------------
# Border Mode Helper
# ----------------------------
def border_mode(name: str):
    mapping = {
        "Constant (black)": cv2.BORDER_CONSTANT,
        "Reflect": cv2.BORDER_REFLECT,
        "Replicate": cv2.BORDER_REPLICATE,
        "Wrap": cv2.BORDER_WRAP,
    }
    return mapping[name]

# ----------------------------
# Manual Color Conversions
# ----------------------------
def rgb_to_gray_manual(img):
    return np.dot(img[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)

def rgb_to_hsv_manual(img):
    img = img.astype("float32") / 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    maxc = np.max(img, axis=-1)
    minc = np.min(img, axis=-1)
    delta = maxc - minc

    hue = np.zeros_like(maxc)
    mask = delta != 0
    hue[mask & (maxc == r)] = (60 * ((g[mask]-b[mask])/delta[mask]) + 360) % 360
    hue[mask & (maxc == g)] = (60 * ((b[mask]-r[mask])/delta[mask]) + 120) % 360
    hue[mask & (maxc == b)] = (60 * ((r[mask]-g[mask])/delta[mask]) + 240) % 360

    sat = np.zeros_like(maxc)
    sat[maxc != 0] = (delta[maxc != 0] / maxc[maxc != 0])

    val = maxc
    hsv = np.stack([hue/2, sat*255, val*255], axis=-1).astype(np.uint8)
    return hsv

def rgb_to_ycbcr_manual(img):
    mat = np.array([[ 0.299,   0.587,   0.114],
                    [-0.1687, -0.3313,  0.5   ],
                    [ 0.5,    -0.4187, -0.0813]])
    shift = np.array([0, 128, 128])
    ycbcr = img @ mat.T + shift
    return np.clip(ycbcr, 0, 255).astype(np.uint8)
