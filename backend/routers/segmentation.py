"""
segmentation.py — Module 3: Image Segmentation
Covers K-Means Clustering, Mean Shift, Seeded Region Growing, and Contour Segmentation.
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


@router.post("/segment")
async def segment_image(
    file: UploadFile = File(...),
    method: str = Form("kmeans"),
    # K-Means parameters
    k_clusters: int = Form(4),
    # Mean Shift parameters
    spatial_radius: int = Form(15),
    color_radius: int = Form(25),
    # Region Growing parameters
    seed_x: Optional[int] = Form(None),
    seed_y: Optional[int] = Form(None),
    region_tolerance: int = Form(20),
):
    """
    Execute Image Segmentation algorithms on an input image.
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
    segmentation_metrics = {}
    method = method.lower()

    if method == "kmeans":
        k = max(2, min(k_clusters, 12))
        pixel_vals = img.reshape((-1, 3))
        pixel_vals = np.float32(pixel_vals)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        processed = segmented_data.reshape(img.shape)

        # Cluster distribution statistics
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_percentages = {
            f"Cluster_{int(lbl)}": round(float(cnt / (h * w)) * 100, 2)
            for lbl, cnt in zip(unique_labels, counts)
        }

        segmentation_metrics = {
            "k_clusters": k,
            "cluster_distribution_pct": cluster_percentages,
        }

    elif method == "meanshift":
        sp = max(5, min(spatial_radius, 40))
        sr = max(5, min(color_radius, 60))
        processed = cv2.pyrMeanShiftFiltering(img, sp=sp, sr=sr, maxLevel=1)
        segmentation_metrics = {
            "spatial_radius": sp,
            "color_radius": sr,
        }

    elif method == "region_growing":
        gray = to_grayscale(img)
        # Default seed point is the center of the image if not supplied
        sx = w // 2 if seed_x is None else max(0, min(seed_x, w - 1))
        sy = h // 2 if seed_y is None else max(0, min(seed_y, h - 1))

        seed_val = float(gray[sy, sx])
        tol = max(5, region_tolerance)

        # 4-connected region growing BFS
        mask = np.zeros((h, w), dtype=np.uint8)
        visited = np.zeros((h, w), dtype=bool)
        queue = [(sy, sx)]
        visited[sy, sx] = True
        mask[sy, sx] = 255

        count = 0
        while queue and count < (h * w):
            cy, cx = queue.pop(0)
            count += 1
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    visited[ny, nx] = True
                    if abs(float(gray[ny, nx]) - seed_val) <= tol:
                        mask[ny, nx] = 255
                        queue.append((ny, nx))

        # Colorize the segmented region with cyan overlay
        colored_mask = img.copy()
        colored_mask[mask > 0] = [6, 182, 212]
        processed = cv2.addWeighted(img, 0.6, colored_mask, 0.4, 0)
        cv2.circle(processed, (sx, sy), 5, (239, 68, 68), -1)  # Red seed point

        segmentation_metrics = {
            "seed_point": [sx, sy],
            "seed_intensity": int(seed_val),
            "region_pixels": count,
            "region_area_pct": round((count / float(h * w)) * 100, 2),
            "tolerance": tol,
        }

    elif method == "contours":
        gray = to_grayscale(img)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        contour_count = len(contours)
        # Draw contours
        cv2.drawContours(processed, contours, -1, (16, 185, 129), 2)

        segmentation_metrics = {
            "contours_found": contour_count,
            "hierarchy_levels": len(hierarchy[0]) if hierarchy is not None else 0,
        }

    else:
        processed = img.copy()

    execution_ms = round((time.time() - t0) * 1000, 2)

    return {
        "status": "success",
        "method": method,
        "execution_time_ms": execution_ms,
        "dimensions": {"width": w, "height": h, "channels": 3},
        "metrics": segmentation_metrics,
        "images": {
            "original": original_b64,
            "processed": encode_image_to_base64(processed),
        },
    }
