/**
 * home.js — Home Page & Architecture Overview.
 */
const HomePage = {
  async render() {
    const app = document.getElementById("app");
    app.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: 32px;">
        <!-- Hero Header -->
        <div class="glass-card" style="position: relative; overflow: hidden; padding: 40px;">
          <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 20px;">
            <div style="max-width: 750px;">
              <div class="badge badge-cyan" style="margin-bottom: 12px;">CSE3010 Computer Vision Course Platform</div>
              <h1 style="font-size: 2.4rem; margin-bottom: 14px; line-height: 1.2;">
                Smart Traffic & Computer Vision <span style="color: var(--accent-cyan);">Video Analytics</span>
              </h1>
              <p style="font-size: 1.05rem;">
                A unified, high-throughput platform demonstrating classical digital image processing, feature extraction, 
                semantic segmentation, deep learning object detection (YOLOv8), multi-object tracking (ByteTrack), and motion flow dynamics.
              </p>
              <div style="display: flex; gap: 14px; margin-top: 24px; flex-wrap: wrap;">
                <button class="btn-primary" onclick="navigate('image')">
                  <span>🖼️ Mode A: Image Analysis</span>
                </button>
                <button class="btn-primary" style="background: linear-gradient(135deg, var(--accent-indigo), #8b5cf6);" onclick="navigate('video')">
                  <span>🎬 Mode B: Video Analytics</span>
                </button>
                <button class="btn-secondary" onclick="HomePage.showArchitectureModal()">
                  <span>🏛️ System Architecture</span>
                </button>
              </div>
            </div>

            <!-- Health Status Widget -->
            <div class="glass-card" style="padding: 16px 22px; background: rgba(8, 14, 26, 0.7); min-width: 220px;">
              <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; font-weight: 700;">Backend Status</div>
              <div id="health-indicator" style="display: flex; align-items: center; gap: 8px; margin-top: 8px; font-weight: 700; font-size: 1.1rem; color: var(--accent-emerald);">
                <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: var(--accent-emerald);"></span>
                Checking…
              </div>
              <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 6px;">FastAPI @ localhost:8000</div>
            </div>
          </div>
        </div>

        <!-- 7 Core Modules Overview Grid -->
        <div>
          <h2 style="font-size: 1.4rem; margin-bottom: 16px;">Comprehensive Algorithmic Modules</h2>
          <div class="grid-3">
            <!-- Module 1 -->
            <div class="glass-card" style="cursor: pointer;" onclick="navigate('image','preprocessing')">
              <div style="font-size: 1.8rem; margin-bottom: 10px;">🔧</div>
              <div class="badge badge-cyan" style="margin-bottom: 8px;">Module 1</div>
              <h3 style="font-size: 1.1rem; margin-bottom: 6px;">Image Preprocessing</h3>
              <p style="font-size: 0.85rem;">Spatial Gaussian/Median filters, Laplacian sharpening, CLAHE, and Mathematical Morphology.</p>
            </div>

            <!-- Module 2 -->
            <div class="glass-card" style="cursor: pointer;" onclick="navigate('image','features')">
              <div style="font-size: 1.8rem; margin-bottom: 10px;">🔍</div>
              <div class="badge badge-cyan" style="margin-bottom: 8px;">Module 2</div>
              <h3 style="font-size: 1.1rem; margin-bottom: 6px;">Feature & Edge Analysis</h3>
              <p style="font-size: 0.85rem;">Canny edge detector, Sobel, Laplacian of Gaussian, Hough lines, Harris corners, SIFT, and HOG.</p>
            </div>

            <!-- Module 3 -->
            <div class="glass-card" style="cursor: pointer;" onclick="navigate('image','segmentation')">
              <div style="font-size: 1.8rem; margin-bottom: 10px;">🎭</div>
              <div class="badge badge-cyan" style="margin-bottom: 8px;">Module 3</div>
              <h3 style="font-size: 1.1rem; margin-bottom: 6px;">Image Segmentation</h3>
              <p style="font-size: 0.85rem;">Unsupervised K-Means clustering, Mean Shift filtering, Seeded Region Growing, and Contours.</p>
            </div>

            <!-- Module 4 -->
            <div class="glass-card" style="cursor: pointer;" onclick="navigate('image','detection')">
              <div style="font-size: 1.8rem; margin-bottom: 10px;">🎯</div>
              <div class="badge badge-emerald" style="margin-bottom: 8px;">Module 4</div>
              <h3 style="font-size: 1.1rem; margin-bottom: 6px;">YOLOv8 Object Detection</h3>
              <p style="font-size: 0.85rem;">Single-shot deep neural network inference across 80 COCO classes with tunable confidence & NMS.</p>
            </div>

            <!-- Module 5 -->
            <div class="glass-card" style="cursor: pointer;" onclick="navigate('video','info')">
              <div style="font-size: 1.8rem; margin-bottom: 10px;">ℹ️</div>
              <div class="badge badge-amber" style="margin-bottom: 8px;">Module 5</div>
              <h3 style="font-size: 1.1rem; margin-bottom: 6px;">Video Telemetry</h3>
              <p style="font-size: 0.85rem;">Container stream decoding (FPS, Resolution, Bitrate) and uniform temporal keyframe sample strips.</p>
            </div>

            <!-- Module 6 -->
            <div class="glass-card" style="cursor: pointer;" onclick="navigate('video','tracking')">
              <div style="font-size: 1.8rem; margin-bottom: 10px;">📡</div>
              <div class="badge badge-amber" style="margin-bottom: 8px;">Module 6</div>
              <h3 style="font-size: 1.1rem; margin-bottom: 6px;">Multi-Object Tracking</h3>
              <p style="font-size: 0.85rem;">YOLOv8 + ByteTrack persistent track IDs with Kalman spatial prediction and category breakdown.</p>
            </div>

            <!-- Module 7 -->
            <div class="glass-card" style="cursor: pointer;" onclick="navigate('video','motion')">
              <div style="font-size: 1.8rem; margin-bottom: 10px;">〰️</div>
              <div class="badge badge-amber" style="margin-bottom: 8px;">Module 7</div>
              <h3 style="font-size: 1.1rem; margin-bottom: 6px;">Motion & Flow Analysis</h3>
              <p style="font-size: 0.85rem;">Dense Farnebäck optical flow (8-bin polar vectors), Lucas-Kanade trails, and MOG2/KNN background subtraction.</p>
            </div>
          </div>
        </div>
      </div>
    `;

    // Ping health endpoint
    try {
      const health = await API.checkHealth();
      const hEl = document.getElementById("health-indicator");
      if (hEl) {
        if (health.status === "healthy") {
          hEl.innerHTML = `<span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: var(--accent-emerald);"></span> Online (7 Modules Active)`;
          hEl.style.color = "var(--accent-emerald)";
        } else {
          hEl.innerHTML = `<span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: var(--accent-rose);"></span> Offline`;
          hEl.style.color = "var(--accent-rose)";
        }
      }
    } catch (e) {
      console.warn("Health check error:", e);
    }
  },

  showArchitectureModal() {
    alert("System Architecture:\\n\\n1. Client Presentation Layer (SPA / Chart.js)\\n2. API Gateway (FastAPI / Asynchronous Routers)\\n3. Service Layer (cv_utils / yolo_service)\\n4. Vision Core Engine (OpenCV / PyTorch / NumPy)\\n5. Media & Buffer Layer (In-memory Base64 Data URIs)");
  }
};
