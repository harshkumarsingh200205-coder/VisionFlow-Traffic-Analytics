"""
feature_analysis.py — Module 2: Feature & Edge Analysis
Covers Differential Gradients, Canny Edges, LoG, Hough Lines, Harris Corners, SIFT, and HOG.
"""
import time
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from skimage.feature import hog
from skimage import exposure

from services.cv_utils import (
    decode_upload,
    encode_image_to_base64,
    resize_for_display,
    to_grayscale,
    ensure_bgr,
)

router = APIRouter()


@router.post("/features")
async def analyze_features(
    file: UploadFile = File(...),
    method: str = Form("canny"),
    # Canny parameters
    canny_low: int = Form(50),
    canny_high: int = Form(150),
    # Hough parameters
    hough_threshold: int = Form(50),
    min_line_length: int = Form(50),
    max_line_gap: int = Form(10),
    # Harris parameters
    harris_block_size: int = Form(2),
    harris_ksize: int = Form(3),
    harris_k: float = Form(0.04),
    harris_thresh: float = Form(0.01),
    # SIFT parameters
    sift_max_features: int = Form(300),
    # Sobel parameters
    sobel_dx: int = Form(1),
    sobel_dy: int = Form(1),
    sobel_ksize: int = Form(3),
):
    """
    Execute Feature Extraction and Edge Analysis on an input image.
    """
    t0 = time.time()
    contents = await file.read()
    img = decode_upload(contents)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    img = resize_for_display(img, max_dim=960)
    original_b64 = encode_image_to_base64(img)
    gray = to_grayscale(img)
    h, w = img.shape[:2]

    processed = img.copy()
    feature_metrics = {}
    method = method.lower()

    if method == "canny":
        edges = cv2.Canny(gray, canny_low, canny_high)
        processed = ensure_bgr(edges)
        edge_pixel_count = int(np.count_nonzero(edges))
        feature_metrics = {
            "edge_pixels": edge_pixel_count,
            "edge_density_pct": round((edge_pixel_count / float(h * w)) * 100, 2),
            "low_threshold": canny_low,
            "high_threshold": canny_high,
        }

    elif method == "sobel":
        ksize = max(1, sobel_ksize if sobel_ksize % 2 == 1 else sobel_ksize + 1)
        if sobel_dx > 0 and sobel_dy == 0:
            grad = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        elif sobel_dx == 0 and sobel_dy > 0:
            grad = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        else:
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            grad = cv2.magnitude(gx, gy)
        grad_abs = cv2.convertScaleAbs(grad)
        processed = ensure_bgr(grad_abs)
        feature_metrics = {
            "max_gradient": float(np.max(grad_abs)),
            "mean_gradient": round(float(np.mean(grad_abs)), 2),
            "ksize": ksize,
        }

    elif method == "laplacian":
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_abs = cv2.convertScaleAbs(lap)
        processed = ensure_bgr(lap_abs)
        feature_metrics = {
            "variance_of_laplacian": round(float(lap.var()), 2),
        }

    elif method == "log":
        # Laplacian of Gaussian
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        log_res = cv2.Laplacian(blurred, cv2.CV_64F)
        log_abs = cv2.convertScaleAbs(log_res)
        processed = ensure_bgr(log_abs)
        feature_metrics = {
            "log_variance": round(float(log_res.var()), 2),
        }

    elif method == "hough_lines":
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )
        line_count = 0
        if lines is not None:
            line_count = len(lines)
            for line in lines:
                pts = line.ravel()
                if len(pts) >= 4:
                    x1, y1, x2, y2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
                    cv2.line(processed, (x1, y1), (x2, y2), (6, 182, 212), 2, cv2.LINE_AA)
        feature_metrics = {
            "lines_detected": line_count,
            "threshold": hough_threshold,
            "min_length": min_line_length,
        }

    elif method == "harris_corner":
        gray_float = np.float32(gray)
        dst = cv2.cornerHarris(
            gray_float,
            blockSize=harris_block_size,
            ksize=harris_ksize if harris_ksize % 2 == 1 else 3,
            k=harris_k,
        )
        # Dilate corner response for visualization
        dst = cv2.dilate(dst, None)
        threshold_val = harris_thresh * dst.max()
        corner_mask = dst > threshold_val
        processed[corner_mask] = [239, 68, 68]  # Highlight in red
        corner_count = int(np.count_nonzero(corner_mask))
        feature_metrics = {
            "corners_detected": corner_count,
            "sensitivity_k": harris_k,
            "max_response": round(float(dst.max()), 4),
        }

    elif method == "sift":
        try:
            sift = cv2.SIFT_create(nfeatures=sift_max_features)
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            processed = cv2.drawKeypoints(
                img,
                keypoints,
                None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                color=(16, 185, 129),
            )
            feature_metrics = {
                "keypoints_detected": len(keypoints) if keypoints else 0,
                "descriptor_dim": 128,
            }
        except Exception as e:
            feature_metrics = {"error": f"SIFT not available: {str(e)}"}

    elif method == "hog":
        try:
            _, hog_image = hog(
                gray,
                orientations=8,
                pixels_per_cell=(16, 16),
                cells_per_block=(1, 1),
                visualize=True,
            )
            hog_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            hog_uint8 = (hog_rescaled * 255).astype("uint8")
            processed = ensure_bgr(hog_uint8)
            feature_metrics = {
                "orientations": 8,
                "pixels_per_cell": "16x16",
            }
        except Exception as e:
            feature_metrics = {"error": f"HOG failed: {str(e)}"}

    else:
        processed = img.copy()

    execution_ms = round((time.time() - t0) * 1000, 2)

    return {
        "status": "success",
        "method": method,
        "execution_time_ms": execution_ms,
        "dimensions": {"width": w, "height": h, "channels": 3},
        "metrics": feature_metrics,
        "images": {
            "original": original_b64,
            "processed": encode_image_to_base64(processed),
        },
    }
