"""
test_video_modules.py — Automated Unit Tests for Mode B (Video Modules 5-7).
"""
import os
import sys
import tempfile
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from main import app

client = TestClient(app)


def create_synthetic_video_bytes(num_frames=20, width=160, height=120, fps=10):
    """Generate a synthetic moving-circle video in an MP4/AVI container."""
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmp:
        tmp_path = tmp.name

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

    for i in range(num_frames):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        # Draw moving circle to generate optical flow and motion
        cx = int(20 + i * (width - 40) / num_frames)
        cy = int(height // 2)
        cv2.circle(frame, (cx, cy), 15, (0, 255, 0), -1)
        out.write(frame)

    out.release()

    with open(tmp_path, "rb") as f:
        data = f.read()

    try:
        os.unlink(tmp_path)
    except Exception:
        pass

    return data


def test_video_info_extraction():
    video_bytes = create_synthetic_video_bytes(num_frames=15)
    response = client.post(
        "/api/video/info",
        files={"file": ("synthetic.avi", video_bytes, "video/x-msvideo")},
        data={"num_keyframes": 4},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "metadata" in data
    assert data["metadata"]["total_frames"] >= 14
    assert len(data["keyframes"]) > 0


def test_video_tracking():
    video_bytes = create_synthetic_video_bytes(num_frames=15)
    response = client.post(
        "/api/video/track",
        files={"file": ("synthetic.avi", video_bytes, "video/x-msvideo")},
        data={"conf": 0.25, "iou": 0.45, "frame_step": 2, "max_output_frames": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "tracking_summary" in data
    assert "sample_frames" in data


def test_video_motion_analysis():
    video_bytes = create_synthetic_video_bytes(num_frames=15)
    for method in ["farneback", "lucas_kanade", "mog2"]:
        response = client.post(
            "/api/video/motion",
            files={"file": ("synthetic.avi", video_bytes, "video/x-msvideo")},
            data={"method": method, "frame_step": 2, "max_output_frames": 3},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "metrics" in data
        assert "sample_frames" in data
