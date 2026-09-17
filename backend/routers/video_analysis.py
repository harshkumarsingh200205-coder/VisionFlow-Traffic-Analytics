"""
video_analysis.py — Module 5: Video Telemetry & Keyframe Sampling
Decodes video stream properties and extracts uniform temporal keyframe sample strips.
"""
import os
import tempfile
import time
import cv2
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from services.cv_utils import encode_image_to_base64, resize_for_display

router = APIRouter()


@router.post("/info")
async def extract_video_info(
    file: UploadFile = File(...),
    num_keyframes: int = Form(8),
):
    """
    Extract video container metadata, stream telemetry, and uniform keyframe strip.
    """
    t0 = time.time()
    contents = await file.read()
    file_size_mb = round(len(contents) / (1024 * 1024), 2)

    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot decode video file stream.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = round(float(cap.get(cv2.CAP_PROP_FPS) or 25.0), 2)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = round(total_frames / fps, 2) if fps > 0 else 0.0

        keyframes = []
        if total_frames > 0:
            k_count = max(2, min(num_keyframes, 16))
            step = max(1, total_frames // k_count)
            sample_indices = [min(i * step, total_frames - 1) for i in range(k_count)]

            for f_idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                small = resize_for_display(frame, max_dim=480)
                # Overlay frame number badge
                cv2.putText(
                    small,
                    f"Frame #{f_idx}",
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (6, 182, 212),
                    1,
                    cv2.LINE_AA,
                )
                keyframes.append({
                    "frame_index": f_idx,
                    "timestamp_sec": round(f_idx / fps, 2) if fps > 0 else 0.0,
                    "image": encode_image_to_base64(small, quality=85),
                })

        cap.release()

    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    execution_ms = round((time.time() - t0) * 1000, 2)

    return {
        "status": "success",
        "execution_time_ms": execution_ms,
        "metadata": {
            "filename": file.filename,
            "resolution": f"{width}x{height}",
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": duration_sec,
            "file_size_mb": file_size_mb,
        },
        "keyframes": keyframes,
    }
