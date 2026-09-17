"""
video_detection.py — Module 6: Multi-Object Detection & Tracking
Runs YOLOv8 + ByteTrack on video frames with temporal decimation and persistent ID telemetry.
"""
import os
import tempfile
import time
from collections import defaultdict
from typing import Dict, Set, List, Any
import cv2
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from services.cv_utils import encode_image_to_base64, resize_for_display
from services import yolo_service

router = APIRouter()


@router.post("/track")
async def track_video_objects(
    file: UploadFile = File(...),
    conf: float = Form(0.35),
    iou: float = Form(0.45),
    frame_step: int = Form(5),
    max_output_frames: int = Form(6),
):
    """
    Run YOLOv8 + ByteTrack multi-object tracking across a video stream.
    Processes every frame_step-th frame for high throughput.
    """
    t0 = time.time()
    contents = await file.read()
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    all_tracks_by_id: Dict[int, Dict[str, Any]] = {}
    by_category: Dict[str, Set[int]] = defaultdict(set)
    sample_frames_b64: List[str] = []

    try:
        yolo_service.reset_tracker()
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot decode video file stream.")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_step = max(1, frame_step)
        sample_interval = max(1, total_frames // (frame_step * max(max_output_frames, 1)))

        frame_idx = 0
        output_count = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % frame_step == 0:
                small = resize_for_display(frame, max_dim=640)
                try:
                    result = yolo_service.track_frame(
                        small, conf_threshold=conf, iou_threshold=iou, persist=True
                    )
                except Exception:
                    frame_idx += 1
                    continue

                for t in result.get("tracks", []):
                    tid = t["track_id"]
                    label = t["label"]
                    confidence = t["conf"]
                    if tid not in all_tracks_by_id:
                        all_tracks_by_id[tid] = {
                            "label": label,
                            "max_conf": confidence,
                            "occurrences": 1,
                        }
                    else:
                        all_tracks_by_id[tid]["max_conf"] = max(
                            all_tracks_by_id[tid]["max_conf"], confidence
                        )
                        all_tracks_by_id[tid]["occurrences"] += 1
                    by_category[label].add(tid)

                # Capture annotated sample frames
                if (
                    output_count < max_output_frames
                    and (frame_idx // frame_step) % sample_interval == 0
                ):
                    annotated = result.get("annotated_frame", small)
                    cv2.putText(
                        annotated,
                        f"Frame {frame_idx} (t={frame_idx/fps:.1f}s)",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (6, 182, 212),
                        2,
                        cv2.LINE_AA,
                    )
                    sample_frames_b64.append(encode_image_to_base64(annotated, quality=85))
                    output_count += 1

            frame_idx += 1

        cap.release()

    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    unique_ids = len(all_tracks_by_id)
    category_summary = {label: len(ids) for label, ids in by_category.items()}
    track_list = [
        {
            "track_id": tid,
            "label": v["label"],
            "max_conf": round(v["max_conf"], 3),
            "frame_detections": v["occurrences"],
        }
        for tid, v in sorted(all_tracks_by_id.items())
    ]

    execution_ms = round((time.time() - t0) * 1000, 2)

    return {
        "status": "success",
        "execution_time_ms": execution_ms,
        "sample_frames": sample_frames_b64,
        "tracking_summary": {
            "unique_objects": unique_ids,
            "frames_processed": frame_idx // frame_step,
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "frame_step": frame_step,
            "by_category": category_summary,
        },
        "track_list": track_list[:60],
    }
