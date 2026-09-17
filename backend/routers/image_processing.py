"""
image_processing.py — Module 1: Preprocessing & Enhancement
Covers Spatial Filters, Histograms, Equalization, Thresholding, and Mathematical Morphology.
"""
import time
from typing import Optional
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from services.cv_utils import (
    decode_upload,
    encode_image_to_base64,
    resize_for_display,
    to_grayscale,
    ensure_bgr,
)

router = APIRouter()


@router.post("/process")
async def process_image(
    file: UploadFile = File(...),
    operation: str = Form("original"),
    # Filter parameters
    kernel_size: int = Form(5),
    sigma: float = Form(1.5),
    # Point transformation parameters
    alpha: float = Form(1.0),
    beta: int = Form(0),
    # Morphology parameters
    morph_op: str = Form("dilation"),
    morph_shape: str = Form("rect"),
    morph_ksize: int = Form(5),
    # Thresholding parameters
    thresh_type: str = Form("otsu"),
    thresh_val: int = Form(127),
):
    """
    Execute Image Preprocessing, Enhancement, or Morphological operations.
    """
    t0 = time.time()
    contents = await file.read()
    img = decode_upload(contents)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    img = resize_for_display(img, max_dim=960)
    original_b64 = encode_image_to_base64(img)
    h, w = img.shape[:2]

    processed = img.copy()
    histogram_data = None
    operation = operation.lower()

    # Kernel validation: ensure odd integer >= 3
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    mk = max(3, morph_ksize if morph_ksize % 2 == 1 else morph_ksize + 1)

    # Structuring element shape
    shape_map = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }
    struct_elem = cv2.getStructuringElement(
        shape_map.get(morph_shape.lower(), cv2.MORPH_RECT), (mk, mk)
    )

    if operation == "original":
        processed = img.copy()

    elif operation == "grayscale":
        gray = to_grayscale(img)
        processed = ensure_bgr(gray)

    elif operation == "gaussian_blur":
        processed = cv2.GaussianBlur(img, (k, k), sigma)

    elif operation == "median_blur":
        processed = cv2.medianBlur(img, k)

    elif operation == "laplacian_sharpen":
        gray = to_grayscale(img)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_abs = cv2.convertScaleAbs(lap)
        lap_bgr = ensure_bgr(lap_abs)
        # High-boost sharpening
        processed = cv2.addWeighted(img, 1.5, lap_bgr, -0.5, 0)

    elif operation == "brightness_contrast":
        # g(x, y) = alpha * f(x, y) + beta
        processed = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    elif operation == "histogram_equalization":
        gray = to_grayscale(img)
        eq = cv2.equalizeHist(gray)
        processed = ensure_bgr(eq)
        # Compute histogram before and after
        hist_orig = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().tolist()
        hist_eq = cv2.calcHist([eq], [0], None, [256], [0, 256]).flatten().tolist()
        histogram_data = {
            "original": [int(x) for x in hist_orig],
            "equalized": [int(x) for x in hist_eq],
        }

    elif operation == "clahe":
        gray = to_grayscale(img)
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe_obj.apply(gray)
        processed = ensure_bgr(eq)
        hist_orig = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().tolist()
        hist_eq = cv2.calcHist([eq], [0], None, [256], [0, 256]).flatten().tolist()
        histogram_data = {
            "original": [int(x) for x in hist_orig],
            "equalized": [int(x) for x in hist_eq],
        }

    elif operation == "threshold":
        gray = to_grayscale(img)
        if thresh_type == "otsu":
            _, th_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif thresh_type == "adaptive_mean":
            th_img = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
            )
        elif thresh_type == "adaptive_gaussian":
            th_img = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        else:
            _, th_img = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        processed = ensure_bgr(th_img)

    elif operation == "morphology":
        gray = to_grayscale(img)
        morph_map = {
            "erosion": cv2.MORPH_ERODE,
            "dilation": cv2.MORPH_DILATE,
            "opening": cv2.MORPH_OPEN,
            "closing": cv2.MORPH_CLOSE,
            "gradient": cv2.MORPH_GRADIENT,
            "tophat": cv2.MORPH_TOPHAT,
            "blackhat": cv2.MORPH_BLACKHAT,
        }
        m_type = morph_map.get(morph_op.lower(), cv2.MORPH_DILATE)
        if m_type == cv2.MORPH_ERODE:
            m_res = cv2.erode(gray, struct_elem, iterations=1)
        elif m_type == cv2.MORPH_DILATE:
            m_res = cv2.dilate(gray, struct_elem, iterations=1)
        else:
            m_res = cv2.morphologyEx(gray, m_type, struct_elem)
        processed = ensure_bgr(m_res)

    else:
        processed = img.copy()

    # Compute intensity distribution if not already computed
    if histogram_data is None:
        gray = to_grayscale(img)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().tolist()
        histogram_data = {"original": [int(x) for x in hist]}

    execution_ms = round((time.time() - t0) * 1000, 2)

    return {
        "status": "success",
        "operation": operation,
        "execution_time_ms": execution_ms,
        "dimensions": {"width": w, "height": h, "channels": 3},
        "histogram": histogram_data,
        "images": {
            "original": original_b64,
            "processed": encode_image_to_base64(processed),
        },
    }
