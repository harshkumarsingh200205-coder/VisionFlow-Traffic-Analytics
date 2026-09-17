"""
yolo_service.py — Deep Learning Object Detection and Multi-Object Tracking Service.
Integrates Ultralytics YOLOv8n with ByteTrack tracking.
"""
import threading
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

# Color palette for object tracking IDs and categories
PALETTE = [
    (6, 182, 212),    # Cyan
    (99, 102, 241),   # Indigo
    (16, 185, 129),   # Emerald
    (245, 158, 11),   # Amber
    (239, 68, 68),    # Red
    (168, 85, 247),   # Purple
    (236, 72, 153),   # Pink
    (20, 184, 166),   # Teal
    (234, 179, 8),    # Yellow
    (59, 130, 246),   # Blue
]

_model = None
_lock = threading.Lock()


def get_model():
    """
    Lazy thread-safe singleton initialization of the YOLOv8n model.
    """
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                try:
                    from ultralytics import YOLO
                    _model = YOLO("yolov8n.pt")
                except Exception as e:
                    print(f"[YOLO Service] Warning: Failed to load Ultralytics YOLO: {e}")
                    _model = None
    return _model


def detect_image(
    img: np.ndarray,
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.45,
) -> Dict[str, Any]:
    """
    Run YOLOv8 object detection on a static image.
    Returns annotated image and detection telemetry.
    """
    model = get_model()
    annotated = img.copy()
    detections: List[Dict[str, Any]] = []
    category_counts: Dict[str, int] = {}

    if model is None:
        return {
            "annotated_frame": annotated,
            "detections": [],
            "total_objects": 0,
            "category_counts": {},
            "status": "model_unavailable",
        }

    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )

    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes
        names = model.names

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                label = names.get(cls_id, f"class_{cls_id}")

                category_counts[label] = category_counts.get(label, 0) + 1
                color = PALETTE[cls_id % len(PALETTE)]

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Draw text badge
                badge = f"{label} {conf:.2f}"
                (tw, th), baseline = cv2.getTextSize(
                    badge, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    annotated,
                    (x1, max(0, y1 - th - baseline - 4)),
                    (x1 + tw + 4, y1),
                    color,
                    -1,
                )
                cv2.putText(
                    annotated,
                    badge,
                    (x1 + 2, max(th + 2, y1 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                detections.append({
                    "label": label,
                    "confidence": round(conf, 3),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                })

    return {
        "annotated_frame": annotated,
        "detections": detections,
        "total_objects": len(detections),
        "category_counts": category_counts,
        "status": "success",
    }


def track_frame(
    img: np.ndarray,
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.45,
    persist: bool = True,
) -> Dict[str, Any]:
    """
    Run YOLOv8 + ByteTrack multi-object tracking on a single video frame.
    Returns annotated frame and active tracks with persistent IDs.
    """
    model = get_model()
    annotated = img.copy()
    active_tracks: List[Dict[str, Any]] = []

    if model is None:
        return {
            "annotated_frame": annotated,
            "tracks": [],
            "status": "model_unavailable",
        }

    results = model.track(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        persist=persist,
        tracker="bytetrack.yaml",
        verbose=False,
    )

    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes
        names = model.names

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                label = names.get(cls_id, f"class_{cls_id}")

                # Track ID
                track_id = int(box.id[0].cpu().numpy()) if box.id is not None else -1
                color = PALETTE[abs(track_id) % len(PALETTE)] if track_id != -1 else PALETTE[cls_id % len(PALETTE)]

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                tid_text = f"#{track_id} " if track_id != -1 else ""
                badge = f"{tid_text}{label} {conf:.2f}"
                (tw, th), baseline = cv2.getTextSize(
                    badge, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    annotated,
                    (x1, max(0, y1 - th - baseline - 4)),
                    (x1 + tw + 4, y1),
                    color,
                    -1,
                )
                cv2.putText(
                    annotated,
                    badge,
                    (x1 + 2, max(th + 2, y1 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                active_tracks.append({
                    "track_id": track_id,
                    "label": label,
                    "conf": round(conf, 3),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                })

    return {
        "annotated_frame": annotated,
        "tracks": active_tracks,
        "status": "success",
    }


def reset_tracker():
    """
    Reset internal tracker state for new video streams.
    """
    model = get_model()
    if model is not None and hasattr(model, "predictor") and model.predictor is not None:
        if hasattr(model.predictor, "trackers"):
            model.predictor.trackers = []
