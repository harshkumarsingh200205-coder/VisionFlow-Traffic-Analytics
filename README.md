# VisionFlow — Smart Traffic & Computer Vision Video Analytics Platform
**CSE3010 Computer Vision | Full-Stack Academic Project**

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python)](https://python.org)
[![OpenCV](https://img.shields.io/badge/CV-OpenCV%204.10-5C3EE8?style=flat&logo=opencv)](https://opencv.org)
[![YOLOv8](https://img.shields.io/badge/Inference-YOLOv8n-FF6F00?style=flat)](https://github.com/ultralytics/ultralytics)
[![Frontend](https://img.shields.io/badge/UI-Dark%20Glassmorphism%20SPA-06B6D4?style=flat)](https://developer.mozilla.org)
[![Course](https://img.shields.io/badge/Course-CSE3010-blue?style=flat)](#academic-alignment)

---

## 1. Project Overview

**VisionFlow** is an interactive, browser-based Computer Vision and Traffic Video Analytics platform developed for academic coursework in **CSE3010 Computer Vision** (VITyarthi Project Framework).

The platform bridges foundational mathematical formulations with empirical computer vision experimentation. It features a dark glassmorphism Single Page Application (SPA) backed by a high-throughput **FastAPI backend** running 30+ algorithms implemented in **OpenCV 4.10, NumPy, scikit-image, and Ultralytics YOLOv8**.

Users can upload static images and video streams, dynamically tune algorithmic hyperparameters in real time via debounced controls, and inspect synchronized side-by-side comparative split views alongside quantitative telemetry dashboards.

---

## 2. Core Functional Modalities & Modules

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                       VisionFlow                                       │
├──────────────────────────────────────────┬─────────────────────────────────────────────┤
│         MODE A: STATIC IMAGE LAB         │          MODE B: VIDEO & TRAFFIC LAB        │
├──────────────────────────────────────────┼─────────────────────────────────────────────┤
│ • Module 1: Preprocessing & Enhancement  │ • Module 5: Video Telemetry & Keyframes     │
│ • Module 2: Feature & Edge Analysis      │ • Module 6: Multi-Object Detection & Track  │
│ • Module 3: Image Segmentation           │ • Module 7: Motion & Optical Flow Analysis  │
│ • Module 4: Deep Learning YOLOv8         │                                             │
└──────────────────────────────────────────┴─────────────────────────────────────────────┘
```

### Mode A: Static Image Processing Lab
1. **Module 1: Image Preprocessing & Enhancement**:
   - Spatial filters: Gaussian Blur ($\sigma$, kernel $3 \dots 31$), Median Blur, Laplacian Sharpening.
   - Histograms: Intensity distribution generation, Global Histogram Equalization, CLAHE.
   - Point transforms: Brightness ($\beta$) and Contrast ($\alpha$) adjustments.
   - Thresholding: Global Otsu, Adaptive Gaussian, and Adaptive Mean thresholding.
   - Mathematical Morphology: Erosion, Dilation, Opening, Closing, Morphological Gradient, Top-Hat, Black-Hat with Rectangular/Elliptic/Cross structuring elements.
2. **Module 2: Feature & Edge Analysis**:
   - Multi-stage Canny Edge Detector with hysteresis thresholding.
   - Differential operators: Sobel 1st-order gradient, Laplacian 2nd-order operator, Laplacian of Gaussian (LoG).
   - Probabilistic Hough Transform line detector (`cv2.HoughLinesP`).
   - Harris Corner Detector with tunable sensitivity $k$.
   - SIFT (Scale-Invariant Feature Transform) and Histogram of Oriented Gradients (HOG).
3. **Module 3: Image Segmentation**:
   - K-Means color space clustering ($k=2 \dots 10$).
   - Mean Shift spatial-color joint domain mode-seeking segmentation.
   - Seeded Region Growing (4-connected flood-fill).
   - Topological contour boundary segmentation.
4. **Module 4: Deep Learning Object Detection**:
   - Ultralytics YOLOv8n anchor-free detection across 80 MS COCO classes.
   - Interactive Confidence ($0.10 \dots 0.95$) and IoU NMS ($0.10 \dots 0.90$) sliders.
   - Category frequency donut charts and bounding box coordinate metadata.

### Mode B: Video & Traffic Analytics Lab
5. **Module 5: Video Telemetry & Keyframe Sampling**:
   - Container metadata decoding: Resolution, FPS, Duration, Total Frames, Bitrate.
   - Uniform temporal keyframe sample strips with frame index timestamps.
6. **Module 6: Multi-Object Detection & Tracking**:
   - YOLOv8 + ByteTrack multi-object tracking with persistent track IDs (`#1, #2...`).
   - Kalman filter spatial prediction and Hungarian bipartite matching.
   - Temporal decimation (`frame_step=1..15`) for high FPS throughput.
   - Active track registry table and class distribution analytics.
7. **Module 7: Motion & Flow Analysis**:
   - Dense Optical Flow (Farnebäck) with HSV velocity mapping and 8-direction polar distribution chart.
   - Sparse Optical Flow (Lucas-Kanade) with historical trajectory trails.
   - Background Subtraction: Gaussian Mixture MOG2 with shadow detection and KNN background subtractor with temporal motion curves.

---

## 3. Directory Structure

```
Smart Traffic Video Analytics Platform/
├── backend/
│   ├── main.py                     # FastAPI entry point, CORS, static mounts
│   ├── requirements.txt            # Python dependencies
│   ├── routers/
│   │   ├── image_processing.py     # Module 1: Preprocessing & morphology
│   │   ├── feature_analysis.py     # Module 2: Edge detection, corners, SIFT, HOG
│   │   ├── segmentation.py         # Module 3: K-Means, Mean Shift, Region Growing
│   │   ├── object_detection.py     # Module 4: YOLOv8 static image inference
│   │   ├── video_analysis.py       # Module 5: Video metadata & keyframe strip
│   │   ├── video_detection.py      # Module 6: ByteTrack tracking & class telemetry
│   │   └── motion_analysis.py      # Module 7: Optical flow & background subtraction
│   └── services/
│       ├── cv_utils.py             # Image encoders, mask overlays, normalizations
│       └── yolo_service.py         # Model singleton & ByteTrack wrapper
├── frontend/
│   ├── index.html                  # Main SPA container & layout
│   ├── css/
│   │   └── styles.css              # Dark Glassmorphism CSS design system
│   └── js/
│       ├── api.js                  # Fetch client & HTTP wrapper
│       ├── app.js                  # Router, navigation state, event dispatcher
│       ├── components/
│       │   ├── dashboard.js        # Chart.js rendering (Histograms, Donuts, Polar)
│       │   └── progressOverlay.js  # Loading overlay
│       └── pages/
│           ├── home.js             # Hero landing, module cards, architecture
│           ├── image_mode.js       # Mode A: 4 image module views & split-view
│           └── video_mode.js       # Mode B: 3 video module views & tracking
├── tests/
│   ├── test_image_modules.py       # Mode A unit tests
│   └── test_video_modules.py       # Mode B unit tests
├── guideline.txt                   # Complete architectural guidelines
├── statement.md                    # Problem statement & academic alignment
└── start.bat                       # 1-click launch automation script
```

---

## 4. Installation & Execution Guide

### Prerequisites
- Python 3.11+
- Modern Web Browser (Chrome, Edge, Firefox, Brave)

### Step 1: Clone & Setup Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 2: 1-Click Launch (Recommended)
Double-click `start.bat` or run:
```powershell
.\start.bat
```

### Step 3: Manual Execution
Alternatively, launch backend and frontend in separate terminals:

**Terminal 1 (Backend API):**
```powershell
cd backend
python main.py
```
*API running at: `http://localhost:8000` (Swagger docs: `http://localhost:8000/api/docs`)*

**Terminal 2 (Frontend Static Server):**
```powershell
cd frontend
python -m http.server 3000
```
*UI accessible at: `http://localhost:3000`*

---

## 5. Automated Testing

Run the automated pytest test suites for Mode A and Mode B:
```powershell
pytest tests/test_image_modules.py -v
pytest tests/test_video_modules.py -v
```

---

## 6. Academic & Syllabus Alignment (CSE3010)

| Course Module / Syllabus Concept | VisionFlow Module Implementation |
|---|---|
| **Module 1: Digital Image Formation & Low Level Processing** | Spatial Convolutions, Gaussian/Median Filters, Histograms, CLAHE (`/api/image/process`) |
| **Module 3: Feature Extraction & Segmentation** | Canny, LoG, Sobel, Hough Lines, Harris, SIFT, HOG, K-Means, Mean Shift, Region Growing (`/api/image/features`, `/api/image/segment`) |
| **Module 4: Pattern Analysis & Motion Analysis** | Dense Farnebäck Optical Flow, Lucas-Kanade KLT, MOG2/KNN Background Subtraction (`/api/video/motion`) |
| **Module 4 / Modern Object Detection** | YOLOv8 Single-Shot Detection & ByteTrack Multi-Object Tracking (`/api/image/detect`, `/api/video/detect`) |
| **Video Telemetry & Frame Decimation** | Video Stream Parsing & Temporal Keyframe Sampling (`/api/video/info`) |
