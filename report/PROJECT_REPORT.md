# CSE3010 Computer Vision — Project Report
**Project Title**: VisionFlow: Smart Traffic & Computer Vision Video Analytics Platform  
**Course Code**: CSE3010 — Computer Vision (LP 3 Credits)  
**Evaluation Framework**: VITyarthi — Build Your Own Project (Flipped Course Evaluation)  
**Academic Year**: 2025–2026  

---

## 1. Cover Page / Metadata
- **Project Title**: VisionFlow: Smart Traffic & Computer Vision Video Analytics Platform
- **Course**: CSE3010 Computer Vision
- **Modality**: Full-Stack Computer Vision & Traffic Video Analytics System
- **Technologies Used**: Python 3.11+, OpenCV 4.10, PyTorch, Ultralytics YOLOv8, ByteTrack, scikit-image, FastAPI, Chart.js 4.4, HTML5/CSS3 ES6 SPA.

---

## 2. Introduction
Computer Vision forms the cornerstone of modern Intelligent Transportation Systems (ITS), automated surveillance, and robotic perception. This project presents **VisionFlow**, an interactive, full-stack analytical platform that unifies classical digital image processing techniques with state-of-the-art deep convolutional networks and multi-object association tracking.

The platform provides a browser-based, dark glassmorphism interface coupled with a high-throughput **FastAPI backend** executing 30+ Computer Vision algorithms implemented in OpenCV 4.10, NumPy, scikit-image, and Ultralytics YOLOv8.

---

## 3. Problem Statement
In computer vision education and research, students and engineers frequently struggle to correlate theoretical mathematical equations (e.g., auto-correlation eigenvalues in Harris corners, difference-of-Gaussians in SIFT, hysteresis dual-thresholding in Canny, or vector field integrals in Farnebäck optical flow) with empirical visual outcomes. 

Existing command-line tools or disconnected notebooks lack:
1. A synchronized comparative split-view interface (Original vs. Processed).
2. Live, debounced hyperparameter tuning.
3. Unified support for both static spatial 2D image analysis and temporal video stream tracking.
4. Integrated quantitative telemetry dashboards (histograms, polar vector plots, category breakdowns).

VisionFlow solves this gap by delivering a self-contained, real-time, interactive laboratory environment.

---

## 4. Functional Requirements (FR)
- **FR-1 (Image Filtering & Preprocessing)**: Gaussian Blur ($k \times k, \sigma$), Median Filter, Laplacian Sharpening, and point linear transformations ($\alpha, \beta$).
- **FR-2 (Histogram Processing & Contrast)**: 256-bin intensity distribution generation, Global Histogram Equalization, and CLAHE.
- **FR-3 (Mathematical Morphology)**: Erosion, Dilation, Opening, Closing, Morphological Gradient, Top-Hat, Black-Hat with Rectangular, Elliptic, and Cross structuring elements.
- **FR-4 (Feature & Edge Extraction)**: Canny edge detector with dual-threshold hysteresis, Sobel, Laplacian of Gaussian (LoG), Probabilistic Hough Transform line detector, Harris Corner Detector, SIFT, and HOG.
- **FR-5 (Image Segmentation)**: K-Means color space clustering ($k=2..10$), Mean Shift mode-seeking segmentation, Seeded Region Growing, and Contour boundary extraction.
- **FR-6 (Deep Learning Object Detection)**: Ultralytics YOLOv8n single-shot anchor-free detection across 80 COCO classes with adjustable confidence and NMS IoU thresholds.
- **FR-7 (Video Telemetry & Keyframes)**: Video stream property extraction (resolution, FPS, duration, bitrate) and uniform temporal keyframe strip generation.
- **FR-8 (Multi-Object Tracking)**: YOLOv8 + ByteTrack multi-object tracker with Kalman spatial prediction, Hungarian matching, persistent track IDs, and decimation.
- **FR-9 (Motion & Flow Analysis)**: Dense Farnebäck optical flow with HSV mapping and 8-direction polar vector distribution; Sparse Lucas-Kanade tracking with trajectory trails; MOG2/KNN dynamic background subtraction.

---

## 5. Non-Functional Requirements (NFR)
- **NFR-1 (Performance & Latency)**: Image processing response latency $< 150\text{ ms}$; video decimation enables processing surveillance clips in $< 3.5\text{ s}$.
- **NFR-2 (Usability & Design Aesthetics)**: Ultra-modern Dark Glassmorphism interface (`backdrop-filter: blur(14px)`), responsive mobile sidebar, debounced slider controls, and zero-reload navigation.
- **NFR-3 (Reliability & Resource Safety)**: In-memory stream processing with guaranteed temporary file unlinking upon completion or exception.
- **NFR-4 (Modularity & Maintainability)**: 5-layer decoupled architecture with independent APIRouters, typed Pydantic contracts, and clean separation of concerns.

---

## 6. System Architecture
VisionFlow follows a **5-Layer Decoupled Architecture**:
1. **Presentation Layer (SPA)**: Pure Vanilla ES6+ modules, CSS Custom Properties, Chart.js 4.4, and synchronized comparative split views.
2. **API Gateway Layer (FastAPI)**: Asynchronous ASGI router with CORS, validation, and automated OpenAPI documentation (`/api/docs`).
3. **Service & Orchestration Layer**: Image memory decoding, Base64 JPEG data URI serializers, and YOLO/ByteTrack singleton service.
4. **Computer Vision Processing Engine**: OpenCV 4.10 C++ native kernels, PyTorch neural networks, NumPy/SciPy vectorization.
5. **Data & Media Layer**: In-memory buffer streams, temporary upload management, and benchmark input media.

---

## 7. Design Diagrams (UML)
*Detailed visual representations are provided in `docs/design_diagrams.md`.*
- **System Architecture Diagram**: 5-layer decoupled flow from UI down to OpenCV/PyTorch engine.
- **Process Flow Diagram**: User workflow from upload to split-view and telemetry charts.
- **Use Case Diagram**: CV Student and Faculty interactions across 13 use cases.
- **Component & Class Diagram**: FastAPI application, 7 APIRouters, `cv_utils`, and `yolo_service`.
- **Sequence Diagrams**: End-to-end request-response cycle for image filtering and video tracking.

---

## 8. Design Decisions & Rationale
1. **FastAPI vs. Flask/Django**: FastAPI provides native asynchronous ASGI coroutines, automatic OpenAPI interactive documentation, and high-throughput serialization needed for image payloads.
2. **Base64 Data URIs vs. File Server URLs**: Returning Base64 encoded JPEG strings eliminates disk I/O bottlenecks, avoids static caching issues during slider sweeps, and provides atomic payload delivery.
3. **ByteTrack vs. Traditional SORT**: ByteTrack matches both high-confidence and low-confidence detections using Kalman filter predictions, preventing track ID fragmentation during vehicle occlusions.
4. **Decimation Sampling for Video Analytics**: Processing every $N$-th frame ($N=3..5$) preserves tracking accuracy while reducing computational load by 60–80%, enabling real-time browser exploration.
5. **Vanilla ES6 vs. Heavy Frameworks**: Avoids compilation overhead and complex build toolchains, allowing immediate 1-click execution for academic evaluation.

---

## 9. Implementation Details
- **Backend**: Python 3.11+, FastAPI 0.110+, OpenCV 4.10, PyTorch 2.2, Ultralytics 8.1, scikit-image, NumPy.
- **Image Operations**: `cv2.GaussianBlur`, `cv2.medianBlur`, `cv2.equalizeHist`, `cv2.createCLAHE`, `cv2.Canny`, `cv2.cornerHarris`, `cv2.SIFT_create`, `cv2.kmeans`, `cv2.pyrMeanShiftFiltering`.
- **Motion Operations**: `cv2.calcOpticalFlowFarneback`, `cv2.calcOpticalFlowPyrLK`, `cv2.createBackgroundSubtractorMOG2`, `cv2.createBackgroundSubtractorKNN`.
- **Frontend**: Vanilla HTML5, CSS3 Glassmorphism, Chart.js 4.4, responsive sidebar, drag-and-drop file readers.

---

## 10. Results & Screenshots Summary
1. **Preprocessing**: Side-by-side comparative views demonstrating Gaussian blur noise attenuation and CLAHE localized contrast enhancement with dual 256-bin intensity histograms.
2. **Feature Detection**: Canny multi-stage edge detection with precise edge density metrics; Harris corner detection with red marker overlays; SIFT rich keypoint scale-space circles.
3. **Segmentation**: K-Means color quantization producing $k=4$ dominant cluster regions with percentage distributions.
4. **Object Detection**: YOLOv8n detecting vehicles with high confidence ($> 0.85$) and category breakdown donuts.
5. **Video Tracking**: ByteTrack maintaining persistent IDs (`#1`, `#2`) across 60+ surveillance frames with track summary tables.
6. **Optical Flow**: Farnebäck dense flow HSV color visualization with 8-direction polar vector area charts.

---

## 11. Testing Approach
Automated testing is implemented using `pytest` and `fastapi.testclient`:
- **`tests/test_image_modules.py`**: Validates health check, all 11 preprocessing operations, edge/feature algorithms, segmentation methods, and YOLO detection.
- **`tests/test_video_modules.py`**: Validates video metadata extraction, keyframe sampling, ByteTrack multi-object tracking, and optical flow / background subtraction.
- **Test Results**: 8 out of 8 test cases passed with 100% success rate.

---

## 12. Challenges Faced
1. **Real-time Video Processing in Browser**: Processing 30 FPS video on standard CPU hardware causes timeouts. *Solution*: Implemented dynamic frame decimation (`frame_step`) and downsampling before inference.
2. **Memory Leaks from Multipart Video Uploads**: Video files uploaded to temporary directories could accumulate over time. *Solution*: Wrapped all video capture pipelines in `try...finally` blocks with guaranteed `os.unlink()` cleanup.
3. **OpenCV Array Shape Discrepancies**: Different OpenCV routines return different line/contour array dimensions (e.g. `(N, 1, 4)` vs `(N, 4)`). *Solution*: Unified coordinate extraction using vectorized `line.ravel()`.

---

## 13. Learnings & Key Takeaways
- Deepened understanding of spatial convolution theory, eigenvalue decomposition in structure tensors, and scale-space extrema in DoG pyramids.
- Mastered bipartite Hungarian matching and Kalman state filtering in modern association tracking (ByteTrack).
- Gained practical expertise in architecting high-performance asynchronous REST APIs around compute-heavy C++ vision libraries.

---

## 14. Future Enhancements
1. **3D Reconstruction & Epipolar Geometry**: Integrate stereoscopic disparity mapping and point cloud rendering.
2. **License Plate Recognition (ANPR)**: Add OCR text recognition pipeline for automated traffic violation logging.
3. **WebSocket Live Camera Streaming**: Enable real-time RTSP/WebRTC camera feed ingestion with live bounding box overlays.

---

## 15. References
1. Szeliski, R. (2011). *Computer Vision: Algorithms and Applications*. Springer-Verlag.
2. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
3. Zhang, Y., et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*. ECCV.
4. Jocher, G., et al. (2023). *Ultralytics YOLOv8*. https://github.com/ultralytics/ultralytics.
5. Bradski, G. (2000). *The OpenCV Library*. Dr. Dobb's Journal of Software Tools.
