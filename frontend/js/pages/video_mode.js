/**
 * video_mode.js — Mode B: Video & Traffic Analytics Lab (Modules 5-7).
 */
const VideoModePage = {
  _currentFile: null,
  _activeTab: "info",

  render(subModule = "info") {
    this._activeTab = subModule;
    const app = document.getElementById("app");
    app.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: 24px;">
        <!-- Header -->
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
          <div>
            <div class="badge badge-amber" style="margin-bottom: 6px;">Mode B Workspace</div>
            <h1 style="font-size: 1.8rem;">Video & Traffic Analytics Lab</h1>
          </div>
          <div>
            <label class="btn-primary" style="background: linear-gradient(135deg, var(--accent-indigo), #8b5cf6); cursor: pointer;">
              <span>📁 Upload Video</span>
              <input type="file" id="video-upload-input" accept="video/*" style="display: none;" onchange="VideoModePage.onFileSelected(event)" />
            </label>
          </div>
        </div>

        <!-- Tab Bar -->
        <div class="tab-bar">
          <button class="tab-btn ${this._activeTab === 'info' ? 'active' : ''}" onclick="VideoModePage.switchTab('info')">5. Video Telemetry & Keyframes</button>
          <button class="tab-btn ${this._activeTab === 'tracking' ? 'active' : ''}" onclick="VideoModePage.switchTab('tracking')">6. YOLOv8 + ByteTrack Tracking</button>
          <button class="tab-btn ${this._activeTab === 'motion' ? 'active' : ''}" onclick="VideoModePage.switchTab('motion')">7. Motion & Flow Analysis</button>
        </div>

        <!-- Parameter Controls Panel -->
        <div class="glass-card" id="video-controls-panel" style="padding: 20px;">
          ${this.renderControlsForTab(this._activeTab)}
        </div>

        <!-- Video Processing Main Output Container -->
        <div id="video-output-container">
          <div class="glass-card" style="text-align: center; padding: 48px 20px;">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">🎬</div>
            <h3 style="margin-bottom: 8px;">No Video Selected</h3>
            <p style="font-size: 0.9rem;">Upload a traffic surveillance video (MP4, AVI, MOV) to execute video analysis pipelines.</p>
          </div>
        </div>
      </div>
    `;
  },

  switchTab(tab) {
    this._activeTab = tab;
    document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
    const ctrlPanel = document.getElementById("video-controls-panel");
    if (ctrlPanel) {
      ctrlPanel.innerHTML = this.renderControlsForTab(tab);
    }
    if (this._currentFile) {
      this.executeCurrentOperation();
    }
  },

  renderControlsForTab(tab) {
    if (tab === "info") {
      return `
        <div class="controls-panel" style="margin-top: 0;">
          <div class="control-group">
            <label class="control-label">Keyframe Strip Sample Count <span id="kcount-val" class="control-value">8</span></label>
            <input type="range" id="num-keyframes" min="4" max="16" step="2" value="8" oninput="document.getElementById('kcount-val').innerText=this.value; VideoModePage.executeCurrentOperation()" />
          </div>
          <button class="btn-primary" onclick="VideoModePage.executeCurrentOperation()">
            <span>▶ Analyze Video Stream</span>
          </button>
        </div>
      `;
    } else if (tab === "tracking") {
      return `
        <div class="controls-panel" style="margin-top: 0;">
          <div class="control-group">
            <label class="control-label">Tracking Model</label>
            <select disabled><option>YOLOv8n + ByteTrack Kalman Filter</option></select>
          </div>
          <div class="control-group">
            <label class="control-label">Confidence Threshold <span id="vconf-val" class="control-value">0.35</span></label>
            <input type="range" id="v-conf" min="0.10" max="0.90" step="0.05" value="0.35" oninput="document.getElementById('vconf-val').innerText=this.value;" />
          </div>
          <div class="control-group">
            <label class="control-label">Frame Decimation Step <span id="vstep-val" class="control-value">5</span></label>
            <input type="range" id="v-step" min="1" max="15" step="1" value="5" oninput="document.getElementById('vstep-val').innerText=this.value;" />
          </div>
          <button class="btn-primary" onclick="VideoModePage.executeCurrentOperation()">
            <span>▶ Run Multi-Object Tracker</span>
          </button>
        </div>
      `;
    } else if (tab === "motion") {
      return `
        <div class="controls-panel" style="margin-top: 0;">
          <div class="control-group">
            <label class="control-label">Motion Algorithm</label>
            <select id="v-motion-method">
              <option value="farneback" selected>Dense Optical Flow (Farnebäck / Polar)</option>
              <option value="lucas_kanade">Sparse Optical Flow (Lucas-Kanade KLT)</option>
              <option value="mog2">Background Subtraction (MOG2 Mixture)</option>
              <option value="knn">Background Subtraction (KNN)</option>
            </select>
          </div>
          <div class="control-group">
            <label class="control-label">Frame Decimation Step <span id="mstep-val" class="control-value">3</span></label>
            <input type="range" id="m-step" min="1" max="10" step="1" value="3" oninput="document.getElementById('mstep-val').innerText=this.value;" />
          </div>
          <button class="btn-primary" onclick="VideoModePage.executeCurrentOperation()">
            <span>▶ Run Motion Analysis</span>
          </button>
        </div>
      `;
    }
  },

  onFileSelected(e) {
    const file = e.target.files[0];
    if (!file) return;
    this._currentFile = file;
    this.executeCurrentOperation();
  },

  async executeCurrentOperation() {
    if (!this._currentFile) return;

    ProgressOverlay.show("Running Video Pipeline…", `Executing ${this._activeTab} analytics on video stream`);
    const formData = new FormData();
    formData.append("file", this._currentFile);

    const outContainer = document.getElementById("video-output-container");

    try {
      if (this._activeTab === "info") {
        formData.append("num_keyframes", document.getElementById("num-keyframes")?.value || "8");
        const res = await API.getVideoInfo(formData);

        if (res && res.metadata) {
          const meta = res.metadata;
          let keyframesHtml = "";
          if (res.keyframes) {
            keyframesHtml = res.keyframes.map(kf => `
              <div class="keyframe-item">
                <img src="${kf.image}" alt="Keyframe ${kf.frame_index}" />
                <div class="keyframe-info">
                  <span>Frame #${kf.frame_index}</span>
                  <span style="color: var(--accent-cyan);">${kf.timestamp_sec}s</span>
                </div>
              </div>
            `).join("");
          }

          outContainer.innerHTML = `
            <div style="display: flex; flex-direction: column; gap: 20px;">
              <!-- Metadata Overview Card -->
              <div class="glass-card" style="padding: 24px;">
                <h3 style="margin-bottom: 16px;">Video Stream Metadata & Telemetry</h3>
                <div class="grid-4">
                  <div class="glass-card" style="padding: 14px; background: rgba(5,10,20,0.5);">
                    <div style="font-size: 0.72rem; color: var(--text-muted);">RESOLUTION</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: var(--accent-cyan); margin-top: 4px;">${meta.resolution}</div>
                  </div>
                  <div class="glass-card" style="padding: 14px; background: rgba(5,10,20,0.5);">
                    <div style="font-size: 0.72rem; color: var(--text-muted);">FRAME RATE</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: var(--accent-emerald); margin-top: 4px;">${meta.fps} FPS</div>
                  </div>
                  <div class="glass-card" style="padding: 14px; background: rgba(5,10,20,0.5);">
                    <div style="font-size: 0.72rem; color: var(--text-muted);">TOTAL FRAMES</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: var(--accent-indigo); margin-top: 4px;">${meta.total_frames}</div>
                  </div>
                  <div class="glass-card" style="padding: 14px; background: rgba(5,10,20,0.5);">
                    <div style="font-size: 0.72rem; color: var(--text-muted);">PLAYBACK DURATION</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: var(--accent-amber); margin-top: 4px;">${meta.duration_sec}s</div>
                  </div>
                </div>
              </div>

              <!-- Temporal Keyframe Sampling Strip -->
              <div class="glass-card" style="padding: 24px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                  <h3>Temporal Keyframe Strip</h3>
                  <span class="badge badge-cyan">${res.keyframes ? res.keyframes.length : 0} Frames Sampled</span>
                </div>
                <div class="keyframe-strip">
                  ${keyframesHtml}
                </div>
              </div>
            </div>
          `;
        }

      } else if (this._activeTab === "tracking") {
        formData.append("conf", document.getElementById("v-conf")?.value || "0.35");
        formData.append("frame_step", document.getElementById("v-step")?.value || "5");
        formData.append("max_output_frames", "6");
        const res = await API.trackVideo(formData);

        if (res && res.tracking_summary) {
          const sum = res.tracking_summary;
          const sampleFramesHtml = res.sample_frames.map(b64 => `
            <div class="keyframe-item" style="min-width: 200px;">
              <img src="${b64}" alt="Track sample" style="height: 140px;" />
            </div>
          `).join("");

          const trackRowsHtml = res.track_list.map(t => `
            <tr>
              <td><span class="badge badge-cyan">#${t.track_id}</span></td>
              <td><strong>${t.label}</strong></td>
              <td>${t.max_conf}</td>
              <td>${t.frame_detections} frames</td>
            </tr>
          `).join("");

          outContainer.innerHTML = `
            <div style="display: flex; flex-direction: column; gap: 20px;">
              <div class="grid-2">
                <div class="glass-card" style="padding: 24px;">
                  <h3 style="margin-bottom: 14px;">Tracking Telemetry Summary</h3>
                  <div style="display: flex; flex-direction: column; gap: 8px;">
                    <div><strong>Unique Tracked Objects:</strong> <span class="badge badge-emerald" style="font-size: 1rem;">${sum.unique_objects}</span></div>
                    <div><strong>Frames Processed:</strong> ${sum.frames_processed} (Decimation step: ${sum.frame_step})</div>
                    <div><strong>Video FPS:</strong> ${sum.fps}</div>
                    <div style="height: 180px; margin-top: 14px; position: relative;">
                      <canvas id="track-category-donut"></canvas>
                    </div>
                  </div>
                </div>

                <div class="glass-card" style="padding: 24px; max-height: 380px; overflow-y: auto;">
                  <h3 style="margin-bottom: 14px;">Active Track Registry</h3>
                  <table class="telemetry-table">
                    <thead>
                      <tr><th>Track ID</th><th>Class</th><th>Max Conf</th><th>Activity</th></tr>
                    </thead>
                    <tbody>${trackRowsHtml}</tbody>
                  </table>
                </div>
              </div>

              <!-- Annotated Frame Samples -->
              <div class="glass-card" style="padding: 24px;">
                <h3 style="margin-bottom: 14px;">ByteTrack Annotated Keyframe Samples</h3>
                <div class="keyframe-strip">${sampleFramesHtml}</div>
              </div>
            </div>
          `;

          if (sum.by_category) {
            Dashboard.renderCategoryDonut("track-category-donut", sum.by_category);
          }
        }

      } else if (this._activeTab === "motion") {
        const method = document.getElementById("v-motion-method")?.value || "farneback";
        formData.append("method", method);
        formData.append("frame_step", document.getElementById("m-step")?.value || "3");
        formData.append("max_output_frames", "6");
        const res = await API.analyzeMotion(formData);

        if (res && res.sample_frames) {
          const sampleFramesHtml = res.sample_frames.map(b64 => `
            <div class="keyframe-item" style="min-width: 200px;">
              <img src="${b64}" alt="Motion sample" style="height: 140px;" />
            </div>
          `).join("");

          outContainer.innerHTML = `
            <div style="display: flex; flex-direction: column; gap: 20px;">
              <div class="grid-2">
                <div class="glass-card" style="padding: 24px;">
                  <h3 style="margin-bottom: 14px;">Motion Velocity & Directional Distribution</h3>
                  <div style="height: 240px; position: relative;">
                    <canvas id="motion-polar-chart"></canvas>
                  </div>
                </div>
                <div class="glass-card" style="padding: 24px;">
                  <h3 style="margin-bottom: 14px;">Motion Telemetry</h3>
                  <div style="display: flex; flex-direction: column; gap: 10px;">
                    <div><strong>Algorithm:</strong> <span class="badge badge-cyan">${res.method.toUpperCase()}</span></div>
                    <div><strong>Execution Time:</strong> ${res.execution_time_ms} ms</div>
                    ${res.metrics && res.metrics.total_motion_energy ? `<div><strong>Total Motion Energy:</strong> ${res.metrics.total_motion_energy}</div>` : ''}
                  </div>
                </div>
              </div>

              <div class="glass-card" style="padding: 24px;">
                <h3 style="margin-bottom: 14px;">Motion Vector Visualizations</h3>
                <div class="keyframe-strip">${sampleFramesHtml}</div>
              </div>
            </div>
          `;

          if (res.metrics && res.metrics.polar_distribution_pct) {
            Dashboard.renderPolarFlow("motion-polar-chart", res.metrics.polar_distribution_pct);
          }
        }
      }
    } catch (err) {
      alert(`Video processing error: ${err.message}`);
    } finally {
      ProgressOverlay.hide();
    }
  }
};
