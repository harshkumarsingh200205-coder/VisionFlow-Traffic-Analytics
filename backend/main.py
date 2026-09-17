"""
main.py — FastAPI Application Entry Point for Smart Traffic Video Analytics Platform.
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routers import (
    image_processing,
    feature_analysis,
    segmentation,
    object_detection,
    video_analysis,
    video_detection,
    motion_analysis,
)

# Ensure temporary uploads directory exists
os.makedirs("uploads", exist_ok=True)

app = FastAPI(
    title="VisionFlow-Traffic-Analytics: Smart Traffic & Computer Vision Analytics API",
    description=(
        "Production-grade REST API for digital image processing, feature extraction, "
        "semantic segmentation, YOLOv8 object detection, ByteTrack tracking, and optical flow."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Enable CORS for local client development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file mount for media
if os.path.exists("uploads"):
    app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Include Modular APIRouters
app.include_router(image_processing.router, prefix="/api/image", tags=["Module 1: Preprocessing & Morphology"])
app.include_router(feature_analysis.router, prefix="/api/image", tags=["Module 2: Feature & Edge Analysis"])
app.include_router(segmentation.router, prefix="/api/image", tags=["Module 3: Image Segmentation"])
app.include_router(object_detection.router, prefix="/api/image", tags=["Module 4: Deep Learning Detection"])
app.include_router(video_analysis.router, prefix="/api/video", tags=["Module 5: Video Telemetry & Keyframes"])
app.include_router(video_detection.router, prefix="/api/video", tags=["Module 6: Multi-Object Tracking"])
app.include_router(motion_analysis.router, prefix="/api/video", tags=["Module 7: Motion & Flow Analysis"])


@app.get("/api/health", tags=["Health"])
def health_check():
    """
    Health check endpoint returning system status and capabilities.
    """
    return {
        "status": "healthy",
        "service": "Smart Traffic Video Analytics API",
        "version": "1.0.0",
        "modules_active": 7,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
