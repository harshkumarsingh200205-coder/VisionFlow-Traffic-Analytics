/**
 * image_mode.js — Mode A: Static Image Processing Lab (Modules 1-4).
 */
const ImageModePage = {
  _currentFile: null,
  _currentDataUrl: null,
  _activeTab: "preprocessing",
  _debounceTimer: null,
  _isProcessing: false,

  render(subModule = "preprocessing") {
    this._activeTab = subModule;
    const app = document.getElementById("app");
    app.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: 24px;">
        <!-- Header -->
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
          <div>
            <div class="badge badge-cyan" style="margin-bottom: 6px;">Mode A Workspace</div>
            <h1 style="font-size: 1.8rem;">Static Image Processing Lab</h1>
          </div>
          <div style="display: flex; gap: 12px; flex-wrap: wrap;">
            <button class="btn-secondary" onclick="ImageModePage.loadSampleImage()">
              <span>⚡ Load Sample Traffic Image</span>
            </button>
            <label class="btn-primary" style="cursor: pointer;">
              <span>📁 Upload Custom Image</span>
              <input type="file" id="img-upload-input" accept="image/*" style="display: none;" onchange="ImageModePage.onFileSelected(event)" />
            </label>
          </div>
        </div>

        <!-- Tab Bar -->
        <div class="tab-bar">
          <button class="tab-btn ${this._activeTab === 'preprocessing' ? 'active' : ''}" onclick="ImageModePage.switchTab('preprocessing')">Preprocessing & Filters</button>
          <button class="tab-btn ${this._activeTab === 'features' ? 'active' : ''}" onclick="ImageModePage.switchTab('features')">Feature & Edge Analysis</button>
          <button class="tab-btn ${this._activeTab === 'segmentation' ? 'active' : ''}" onclick="ImageModePage.switchTab('segmentation')">Image Segmentation</button>
          <button class="tab-btn ${this._activeTab === 'detection' ? 'active' : ''}" onclick="ImageModePage.switchTab('detection')">YOLOv8 Detection</button>
        </div>

        <!-- Parameter Controls Panel -->
        <div class="glass-card" id="image-controls-panel" style="padding: 20px;">
          ${this.renderControlsForTab(this._activeTab)}
        </div>

        <!-- Split-View Comparative Display -->
        <div class="split-view-container">
          <!-- Original Image Pane -->
          <div class="image-pane">
            <div class="pane-header">
              <span>Original Source Image</span>
              <span id="orig-dim-badge" class="badge badge-cyan">${this._currentFile ? 'Loaded' : 'No Image'}</span>
            </div>
            <div class="pane-body">
              <img id="img-original" src="${this._currentDataUrl || ''}" alt="Original" style="${this._currentDataUrl ? 'display: block;' : 'display: none;'}" />
              <div id="orig-placeholder" style="${this._currentDataUrl ? 'display: none;' : 'color: var(--text-muted); font-size: 0.9rem;'}">Please upload an image or click 'Load Sample Traffic Image'</div>
            </div>
          </div>

          <!-- Processed Output Pane -->
          <div class="image-pane">
            <div class="pane-header">
              <span>Processed Output Image</span>
              <span id="proc-exec-badge" class="badge badge-emerald">Ready</span>
            </div>
            <div class="pane-body">
              <img id="img-processed" src="" alt="Processed output" style="display: none;" />
              <div id="proc-placeholder" style="color: var(--text-muted); font-size: 0.9rem;">Algorithmic output will appear here</div>
            </div>
          </div>
        </div>

        <!-- Dynamic Visual Analytics Dashboard -->
        <div class="glass-card" id="image-analytics-panel" style="padding: 20px; display: none;">
          <h3 style="font-size: 1.1rem; margin-bottom: 14px;">Quantitative Analytical Metrics</h3>
          <div class="grid-2">
            <div style="height: 220px; position: relative;">
              <canvas id="img-histogram-chart"></canvas>
            </div>
            <div id="img-metrics-container" style="display: flex; flex-direction: column; justify-content: center; gap: 8px;">
              <!-- Dynamic Metric Badges -->
            </div>
          </div>
        </div>
      </div>
    `;

    if (this._currentFile) {
      this.debouncedExecute();
    }
  },

  switchTab(tab) {
    this._activeTab = tab;
    document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
    const ctrlPanel = document.getElementById("image-controls-panel");
    if (ctrlPanel) {
      ctrlPanel.innerHTML = this.renderControlsForTab(tab);
    }
    if (this._currentFile) {
      this.debouncedExecute();
    }
  },

  renderControlsForTab(tab) {
    if (tab === "preprocessing") {
      return `
        <div class="controls-panel" style="margin-top: 0;">
          <div class="control-group">
            <label class="control-label">Algorithm Operation</label>
            <select id="prep-op" onchange="ImageModePage.debouncedExecute()">
              <option value="original">Original (Pass-through)</option>
              <option value="grayscale">Grayscale Conversion</option>
              <option value="gaussian_blur" selected>Gaussian Smoothing Filter</option>
              <option value="median_blur">Median Noise Filter</option>
              <option value="laplacian_sharpen">Laplacian Sharpening</option>
              <option value="clahe">CLAHE Contrast Enhancement</option>
              <option value="histogram_equalization">Global Histogram Equalization</option>
              <option value="brightness_contrast">Brightness & Contrast</option>
              <option value="threshold">Binary Thresholding (Otsu)</option>
              <option value="morphology">Mathematical Morphology</option>
            </select>
          </div>
          <div class="control-group">
            <label class="control-label">Kernel Dimension <span id="ksize-val" class="control-value">5</span></label>
            <input type="range" id="prep-ksize" min="3" max="31" step="2" value="5" oninput="document.getElementById('ksize-val').innerText=this.value; ImageModePage.debouncedExecute()" />
          </div>
          <div class="control-group">
            <label class="control-label">Gaussian Sigma (σ) <span id="sigma-val" class="control-value">1.5</span></label>
            <input type="range" id="prep-sigma" min="0.5" max="10.0" step="0.5" value="1.5" oninput="document.getElementById('sigma-val').innerText=this.value; ImageModePage.debouncedExecute()" />
          </div>
          <div class="control-group">
            <label class="control-label">Contrast Multiplier (α) <span id="alpha-val" class="control-value">1.0</span></label>
            <input type="range" id="prep-alpha" min="0.2" max="3.0" step="0.1" value="1.0" oninput="document.getElementById('alpha-val').innerText=this.value; ImageModePage.debouncedExecute()" />
          </div>
          <div class="control-group">
            <label class="control-label">Brightness Shift (β) <span id="beta-val" class="control-value">0</span></label>
            <input type="range" id="prep-beta" min="-100" max="100" step="5" value="0" oninput="document.getElementById('beta-val').innerText=this.value; ImageModePage.debouncedExecute()" />
          </div>
        </div>
      `;
    } else if (tab === "features") {
      return `
        <div class="controls-panel" style="margin-top: 0;">
          <div class="control-group">
            <label class="control-label">Feature Method</label>
            <select id="feat-method" onchange="ImageModePage.debouncedExecute()">
              <option value="canny" selected>Canny Multi-Stage Edge Detector</option>
              <option value="sobel">Sobel 1st-Order Derivative</option>
              <option value="laplacian">Laplacian 2nd-Order Operator</option>
              <option value="log">Laplacian of Gaussian (LoG)</option>
              <option value="hough_lines">Probabilistic Hough Lines</option>
              <option value="harris_corner">Harris Corner Detector</option>
              <option value="sift">SIFT Keypoint Descriptors</option>
              <option value="hog">Histogram of Oriented Gradients (HOG)</option>
            </select>
          </div>
          <div class="control-group">
            <label class="control-label">Canny Low Threshold <span id="clow-val" class="control-value">50</span></label>
            <input type="range" id="canny-low" min="10" max="250" step="5" value="50" oninput="document.getElementById('clow-val').innerText=this.value; ImageModePage.debouncedExecute()" />
          </div>
          <div class="control-group">
            <label class="control-label">Canny High Threshold <span id="chigh-val" class="control-value">150</span></label>
            <input type="range" id="canny-high" min="20" max="300" step="5" value="150" oninput="document.getElementById('chigh-val').innerText=this.value; ImageModePage.debouncedExecute()" />
          </div>
          <div class="control-group">
            <label class="control-label">Harris Sensitivity (k) <span id="hk-val" class="control-value">0.04</span></label>
            <input type="range" id="harris-k" min="0.01" max="0.10" step="0.01" value="0.04" oninput="document.getElementById('hk-val').innerText=this.value; ImageModePage.debouncedExecute()" />
          </div>
        </div>
      `;
    } else if (tab === "segmentation") {
      return `
        <div class="controls-panel" style="margin-top: 0;">
          <div class="control-group">
            <label class="control-label">Segmentation Algorithm</label>
            <select id="seg-method" onchange="ImageModePage.debouncedExecute()">
              <option value="kmeans" selected>K-Means Color Space Clustering</option>
              <option value="meanshift">Mean Shift Joint Mode-Seeking</option>
              <option value="region_growing">Seeded Region Growing</option>
              <option value="contours">Hierarchical Contour Segmentation</option>
            </select>
          </div>
          <div class="control-group">
            <label class="control-label">Clusters (k) <span id="kclusters-val" class="control-value">4</span></label>
            <input type="range" id="k-clusters" min="2" max="10" step="1" value="4" oninput="document.getElementById('kclusters-val').innerText=this.value; ImageModePage.debouncedExecute()" />
          </div>
          <div class="control-group">
            <label class="control-label">Region Growth Tolerance <span id="tol-val" class="control-value">20</span></label>
            <input type="range" id="region-tol" min="5" max="60" step="5" value="20" oninput="document.getElementById('tol-val').innerText=this.value; ImageModePage.debouncedExecute()" />
          </div>
        </div>
      `;
    } else if (tab === "detection") {
      return `
        <div class="controls-panel" style="margin-top: 0;">
          <div class="control-group">
            <label class="control-label">Model Architecture</label>
            <select disabled><option>Ultralytics YOLOv8n (MS COCO 80 Classes)</option></select>
          </div>
          <div class="control-group">
            <label class="control-label">Confidence Threshold <span id="conf-val" class="control-value">0.35</span></label>
            <input type="range" id="yolo-conf" min="0.10" max="0.95" step="0.05" value="0.35" oninput="document.getElementById('conf-val').innerText=this.value; ImageModePage.debouncedExecute()" />
          </div>
          <div class="control-group">
            <label class="control-label">NMS IoU Threshold <span id="iou-val" class="control-value">0.45</span></label>
            <input type="range" id="yolo-iou" min="0.10" max="0.90" step="0.05" value="0.45" oninput="document.getElementById('iou-val').innerText=this.value; ImageModePage.debouncedExecute()" />
          </div>
        </div>
      `;
    }
  },

  async loadSampleImage() {
    try {
      ProgressOverlay.show("Loading Sample Asset…", "Generating synthetic traffic image");
      // Create sample image via HTML5 Canvas
      const canvas = document.createElement("canvas");
      canvas.width = 1280;
      canvas.height = 720;
      const ctx = canvas.getContext("2d");

      // Road background
      ctx.fillStyle = "#1e2430";
      ctx.fillRect(0, 0, 1280, 720);
      ctx.fillStyle = "#334155";
      ctx.fillRect(0, 350, 1280, 370);

      // Lane dividers
      ctx.fillStyle = "#e2e8f0";
      for (let x = 40; x < 1280; x += 160) {
        ctx.fillRect(x, 520, 90, 15);
      }

      // Vehicles
      ctx.fillStyle = "#0284c7";
      ctx.fillRect(220, 420, 220, 130);
      ctx.fillStyle = "#fff";
      ctx.font = "bold 24px Inter";
      ctx.fillText("CAR", 300, 490);

      ctx.fillStyle = "#16a34a";
      ctx.fillRect(660, 370, 300, 200);
      ctx.fillStyle = "#fff";
      ctx.fillText("BUS", 780, 480);

      canvas.toBlob((blob) => {
        if (!blob) return;
        const file = new File([blob], "sample_traffic.jpg", { type: "image/jpeg" });
        this._currentFile = file;
        this._currentDataUrl = canvas.toDataURL("image/jpeg");

        const origImg = document.getElementById("img-original");
        const origPl = document.getElementById("orig-placeholder");
        if (origImg && origPl) {
          origImg.src = this._currentDataUrl;
          origImg.style.display = "block";
          origPl.style.display = "none";
        }
        ProgressOverlay.hide();
        this.executeCurrentOperation();
      }, "image/jpeg", 0.95);

    } catch (e) {
      ProgressOverlay.hide();
      alert("Failed to create sample image: " + e.message);
    }
  },

  onFileSelected(e) {
    const file = e.target.files[0];
    if (!file) return;
    this._currentFile = file;

    const reader = new FileReader();
    reader.onload = (event) => {
      this._currentDataUrl = event.target.result;
      const origImg = document.getElementById("img-original");
      const origPl = document.getElementById("orig-placeholder");
      if (origImg && origPl) {
        origImg.src = this._currentDataUrl;
        origImg.style.display = "block";
        origPl.style.display = "none";
      }
      this.executeCurrentOperation();
    };
    reader.readAsDataURL(file);
  },

  debouncedExecute() {
    clearTimeout(this._debounceTimer);
    this._debounceTimer = setTimeout(() => {
      this.executeCurrentOperation();
    }, 180);
  },

  async executeCurrentOperation() {
    if (!this._currentFile || this._isProcessing) return;

    this._isProcessing = true;
    ProgressOverlay.show("Running Computer Vision Pipeline…", `Executing ${this._activeTab} algorithm`);
    
    // Ensure clean Blob/File payload
    const formData = new FormData();
    formData.append("file", this._currentFile, this._currentFile.name || "image.jpg");

    try {
      let res;
      if (this._activeTab === "preprocessing") {
        formData.append("operation", document.getElementById("prep-op")?.value || "gaussian_blur");
        formData.append("kernel_size", document.getElementById("prep-ksize")?.value || "5");
        formData.append("sigma", document.getElementById("prep-sigma")?.value || "1.5");
        formData.append("alpha", document.getElementById("prep-alpha")?.value || "1.0");
        formData.append("beta", document.getElementById("prep-beta")?.value || "0");
        res = await API.processImage(formData);
      } else if (this._activeTab === "features") {
        formData.append("method", document.getElementById("feat-method")?.value || "canny");
        formData.append("canny_low", document.getElementById("canny-low")?.value || "50");
        formData.append("canny_high", document.getElementById("canny-high")?.value || "150");
        formData.append("harris_k", document.getElementById("harris-k")?.value || "0.04");
        res = await API.analyzeFeatures(formData);
      } else if (this._activeTab === "segmentation") {
        formData.append("method", document.getElementById("seg-method")?.value || "kmeans");
        formData.append("k_clusters", document.getElementById("k-clusters")?.value || "4");
        formData.append("region_tolerance", document.getElementById("region-tol")?.value || "20");
        res = await API.segmentImage(formData);
      } else if (this._activeTab === "detection") {
        formData.append("conf_threshold", document.getElementById("yolo-conf")?.value || "0.35");
        formData.append("iou_threshold", document.getElementById("yolo-iou")?.value || "0.45");
        res = await API.detectObjects(formData);
      }

      if (res && res.images && res.images.processed) {
        const procImg = document.getElementById("img-processed");
        const procPl = document.getElementById("proc-placeholder");
        const execBadge = document.getElementById("proc-exec-badge");
        const dimBadge = document.getElementById("orig-dim-badge");

        if (procImg && procPl) {
          procImg.src = res.images.processed;
          procImg.style.display = "block";
          procPl.style.display = "none";
        }
        if (execBadge) {
          execBadge.innerText = `${res.execution_time_ms} ms`;
        }
        if (dimBadge && res.dimensions) {
          dimBadge.innerText = `${res.dimensions.width}x${res.dimensions.height}`;
        }

        // Render Analytics Panel if data available
        const analyticsPanel = document.getElementById("image-analytics-panel");
        const metricsContainer = document.getElementById("img-metrics-container");
        if (analyticsPanel && metricsContainer) {
          analyticsPanel.style.display = "block";
          if (res.histogram) {
            Dashboard.renderHistogram("img-histogram-chart", res.histogram);
          } else if (res.category_counts) {
            Dashboard.renderCategoryDonut("img-histogram-chart", res.category_counts);
          }

          let metricsHtml = `<div><strong>Execution Latency:</strong> <span class="badge badge-cyan">${res.execution_time_ms} ms</span></div>`;
          if (res.total_objects !== undefined) {
            metricsHtml += `<div><strong>Total Detected Objects:</strong> <span class="badge badge-emerald">${res.total_objects}</span></div>`;
          }
          if (res.metrics) {
            for (const [k, v] of Object.entries(res.metrics)) {
              metricsHtml += `<div><strong>${k.replace(/_/g, " ").toUpperCase()}:</strong> ${JSON.stringify(v)}</div>`;
            }
          }
          metricsContainer.innerHTML = metricsHtml;
        }
      }
    } catch (err) {
      alert(`Processing error: ${err.message}`);
    } finally {
      this._isProcessing = false;
      ProgressOverlay.hide();
    }
  }
};
