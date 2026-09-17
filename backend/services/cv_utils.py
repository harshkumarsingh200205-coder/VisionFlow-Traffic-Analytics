"""
cv_utils.py — Shared OpenCV and image processing helper utilities for the Smart Traffic Analytics platform.
"""
import base64
import io
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image


def encode_image_to_base64(img: np.ndarray, quality: int = 92) -> str:
    """
    Encode an OpenCV image (BGR or grayscale) to a Base64 JPEG data URI string.
    """
    if img is None or img.size == 0:
        return ""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    success, buffer = cv2.imencode(".jpg", img, encode_param)
    if not success:
        return ""
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def decode_upload(file_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decode raw multipart uploaded bytes into an OpenCV BGR numpy array.
    Uses cv2.imdecode with robust PIL fallback for all image formats (PNG/Alpha, WebP, AVIF, TIFF, etc.).
    """
    if not file_bytes or len(file_bytes) == 0:
        return None

    # First attempt: OpenCV native buffer decode
    try:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None and img.size > 0:
            return img
    except Exception:
        pass

    # Fallback attempt: Pillow (handles RGBA, WebP, TIFF, Animated GIF 1st frame, etc.)
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        rgb_arr = np.array(pil_img)
        bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        return bgr_arr
    except Exception as e:
        print(f"[decode_upload] Error decoding image buffer: {e}")
        return None


def resize_for_display(img: np.ndarray, max_dim: int = 960) -> np.ndarray:
    """
    Resize image so its longest dimension does not exceed max_dim, preserving aspect ratio.
    """
    if img is None or img.size == 0:
        return img
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / float(max(h, w))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    """
    Ensure the image is a 3-channel BGR image.
    """
    if img is None:
        return img
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if len(img.shape) == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert an image to a single-channel grayscale image.
    """
    if img is None:
        return img
    if len(img.shape) == 2:
        return img
    if len(img.shape) == 3 and img.shape[2] >= 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def apply_colormap_jet(gray_img: np.ndarray) -> np.ndarray:
    """
    Apply JET colormap to a grayscale intensity image for pseudo-color thermal visualization.
    """
    if gray_img is None or gray_img.size == 0:
        return gray_img
    if len(gray_img.shape) == 3:
        gray_img = to_grayscale(gray_img)
    norm = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


def overlay_mask(
    img: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Overlay a binary boolean mask onto a BGR image with alpha transparency.
    """
    if img is None or mask is None:
        return img
    bgr = ensure_bgr(img)
    result = bgr.copy()
    colored = np.zeros_like(bgr)
    colored[:] = color
    mask_bool = mask > 0
    if len(mask_bool.shape) == 2 and len(result.shape) == 3:
        mask_bool_3d = np.repeat(mask_bool[:, :, np.newaxis], 3, axis=2)
        blended = cv2.addWeighted(bgr, 1.0 - alpha, colored, alpha, 0)
        result[mask_bool_3d] = blended[mask_bool_3d]
    return result


def draw_text_with_bg(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font_scale: float = 0.55,
    thickness: int = 1,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (20, 20, 20),
) -> np.ndarray:
    """
    Draw text with a high-contrast filled background rectangle for readability.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(
        img,
        (x - 2, y - th - baseline - 2),
        (x + tw + 4, y + baseline),
        bg_color,
        -1,
    )
    cv2.putText(
        img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA
    )
    return img
