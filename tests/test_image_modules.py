"""
test_image_modules.py — Automated Unit Tests for Mode A (Image Processing Modules 1-4).
"""
import io
import sys
import os
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Add backend directory to sys.path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from main import app

client = TestClient(app)


def create_test_image_bytes(width=200, height=150, color=(100, 150, 200)):
    """Generate a synthetic test image as JPEG bytes."""
    img = np.full((height, width, 3), color, dtype=np.uint8)
    # Add a rectangle and circle for edge and feature detection tests
    cv2.rectangle(img, (30, 30), (100, 100), (255, 255, 255), -1)
    cv2.circle(img, (150, 80), 25, (0, 0, 255), -1)
    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["modules_active"] == 7


def test_image_preprocessing_filters():
    img_bytes = create_test_image_bytes()
    for op in ["grayscale", "gaussian_blur", "median_blur", "laplacian_sharpen", "clahe", "threshold"]:
        response = client.post(
            "/api/image/process",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            data={"operation": op, "kernel_size": 5, "sigma": 1.2},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "images" in data
        assert data["images"]["processed"].startswith("data:image/jpeg;base64,")


def test_feature_extraction():
    img_bytes = create_test_image_bytes()
    for method in ["canny", "sobel", "harris_corner", "hough_lines", "log"]:
        response = client.post(
            "/api/image/features",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            data={"method": method, "canny_low": 50, "canny_high": 150},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "metrics" in data


def test_image_segmentation():
    img_bytes = create_test_image_bytes()
    for method in ["kmeans", "region_growing", "contours"]:
        response = client.post(
            "/api/image/segment",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            data={"method": method, "k_clusters": 3},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "metrics" in data


def test_object_detection():
    img_bytes = create_test_image_bytes()
    response = client.post(
        "/api/image/detect",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"conf_threshold": 0.25, "iou_threshold": 0.45},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "total_objects" in data
    assert "detections" in data
