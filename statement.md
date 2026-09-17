# Problem Statement & Academic Alignment Specification

## Course: CSE3010 Computer Vision (LP 3 Credits)
**Project Title**: Smart Traffic & Computer Vision Video Analytics Platform  
**Evaluation Program**: VITyarthi - Build Your Own Project

---

## 1. Problem Statement

Modern intelligent transportation systems (ITS), smart city surveillance networks, and academic computer vision research require robust, end-to-end tooling to analyze spatial image structures and dynamic temporal video streams. While classical algorithms (edge detection, spatial convolution filters, mathematical morphology, morphological segmentation, and optical flow) form the foundational mathematics of computer vision, modern workflows rely heavily on deep neural networks (single-shot detectors like YOLO) and association trackers (ByteTrack).

Students, researchers, and engineers frequently encounter a disconnect between theoretical mathematical formulas (such as eigenvalues in Harris corners, difference-of-Gaussians in SIFT, hysteresis thresholds in Canny, or vector fields in Farnebäck optical flow) and their real-world visual manifestations. Existing command-line scripts or disconnected notebooks lack an intuitive, synchronized comparative interface allowing real-time parameter tuning, visual telemetry dashboards, and multi-modal analysis in a single unified platform.

---

## 2. Project Objectives

1. **Unify Classical & Modern Vision**: Bridge foundational digital image processing (filtering, histograms, morphology, edges, segmentation) with modern deep learning object detection (YOLOv8) and multi-object tracking (ByteTrack).
2. **Empirical Parameter Sweeping**: Provide interactive UI controls (sliders, matrix dropdowns, kernel dimension selectors) enabling users to observe instantaneous algorithmic transformations and failure modes.
3. **Synchronized Comparative Analysis**: Deliver side-by-side comparative split views (Original vs. Processed) to quantitatively and visually measure contrast improvements, edge localization, and tracking persistence.
4. **Rich Quantitative Dashboards**: Present real-time histograms, category frequency donut charts, and 8-direction polar motion vector distributions.
5. **Production-Ready Asynchronous Architecture**: Construct an asynchronous REST API backend with automated OpenAPI documentation paired with a dark glassmorphic Single Page Application.

---

## 3. Scope of the Project

### In Scope
- **Mode A: Static Image Processing Lab**
  - Module 1: Preprocessing & Enhancement (Gaussian, Median, Laplacian, Histograms, CLAHE, Morphology).
  - Module 2: Feature & Edge Analysis (Sobel, Canny, LoG, Hough Lines, Harris Corners, SIFT, HOG).
  - Module 3: Image Segmentation (K-Means color clustering, Mean Shift, Seeded Region Growing, Contour Hierarchy).
  - Module 4: Deep Learning Object Detection (YOLOv8n COCO 80 classes with tunable confidence and NMS IoU).
- **Mode B: Video & Traffic Analytics Lab**
  - Module 5: Video Telemetry & Temporal Keyframe Sampling (Stream decoding, FPS, resolution, timeline strips).
  - Module 6: Multi-Object Detection & Tracking (YOLOv8 + ByteTrack persistent track IDs, vehicle counts).
  - Module 7: Motion & Flow Analysis (Dense Farnebäck flow with HSV mapping & 8-bin polar distribution, Sparse Lucas-Kanade with trajectory trails, MOG2/KNN background subtraction).

### Out of Scope
- Generative AI synthesis (GANs, Diffusion models).
- Multi-tenant cloud databases and authentication tokens (designed for zero-friction local execution).

---

## 4. Target Users

1. **Computer Vision Students**: Validating lecture concepts, understanding mathematical parameter sensitivity, and gathering visual artifacts for laboratory portfolios.
2. **Faculty & Lab Instructors**: Conducting live classroom demonstrations and evaluating student viva practicals with instant parameter adjustments.
3. **Applied AI Engineers & Researchers**: Rapidly prototyping and benchmarking preprocessing filters and tracking pipelines on novel surveillance datasets.

---

## 5. Functional & Non-Functional Requirements

### 5.1 Functional Requirements (FR)
- **FR-1**: Support multi-format image ingestion (JPEG, PNG, WebP) and video streams (MP4, AVI, MOV, MKV).
- **FR-2**: Provide 11+ image preprocessing filters, histogram equalizers, and mathematical morphological operators.
- **FR-3**: Extract multi-order gradients, Canny edges, geometric lines, Harris corners, SIFT keypoints, and HOG features.
- **FR-4**: Perform unsupervised K-Means and Mean Shift clustering alongside seeded region growing.
- **FR-5**: Execute YOLOv8 inference with adjustable confidence thresholds and NMS IoU thresholds.
- **FR-6**: Decode video metadata and generate uniform temporal keyframe sample strips.
- **FR-7**: Track objects across video frames with persistent IDs and category breakdown metrics.
- **FR-8**: Calculate dense Farnebäck flow, sparse Lucas-Kanade tracking, and MOG2/KNN background segmentation.

### 5.2 Non-Functional Requirements (NFR)
- **NFR-1 (Performance)**: Image processing response latency $< 150\text{ ms}$ for standard inputs; video decimation enables multi-frame tracking in $< 3\text{ s}$.
- **NFR-2 (Usability)**: Responsive Dark Glassmorphism interface with debounced sliders, split-view comparisons, and zero reload requirements.
- **NFR-3 (Reliability & Robustness)**: In-memory stream processing with guaranteed temporary file cleanup upon completion or error.
- **NFR-4 (Maintainability & Modularity)**: Decoupled 5-layer architecture with independent APIRouters and typed Pydantic contracts.
