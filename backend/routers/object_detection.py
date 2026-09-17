"""
object_detection.py — Module 4: Deep Learning Object Detection
Executes YOLOv8n single-shot inference on static images with adjustable confidence and NMS IoU.
"""
import time
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from services.cv_utils import (
    decode_upload,
    encode_image_to_base64,
    resize_for_display,
)
from services import yolo_service

router = APIRouter()


@router.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.35),
    iou_threshold: float = Form(0.45),
):
    """
    Run YOLOv8 object detection on a static uploaded image.
    """
    t0 = time.time()
    contents = await file.read()
    img = decode_upload(contents)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    img = resize_for_display(img, max_dim=960)
    original_b64 = encode_image_to_base64(img)
    h, w = img.shape[:2]

    # Run YOLO detection service
    result = yolo_service.detect_image(
        img, conf_threshold=conf_threshold, iou_threshold=iou_threshold
    )

    annotated_img = result["annotated_frame"]
    detections = result["detections"]
    category_counts = result["category_counts"]
    total_objects = result["total_objects"]

    execution_ms = round((time.time() - t0) * 1000, 2)

    return {
        "status": "success",
        "execution_time_ms": execution_ms,
        "dimensions": {"width": w, "height": h, "channels": 3},
        "total_objects": total_objects,
        "category_counts": category_counts,
        "detections": detections,
        "hyperparameters": {
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
        },
        "images": {
            "original": original_b64,
            "processed": encode_image_to_base64(annotated_img),
        },
    }
