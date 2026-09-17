/**
 * api.js — Reusable API Service Client for VisionFlow Platform.
 */
const API_BASE = "http://localhost:8000/api";

const API = {
  /**
   * Check backend health status.
   */
  async checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      return await res.json();
    } catch (err) {
      console.error("[API] Health check failed:", err);
      return { status: "offline", error: err.message };
    }
  },

  /**
   * Module 1: Preprocessing & Enhancement
   */
  async processImage(formData) {
    return this._postMultipart(`${API_BASE}/image/process`, formData);
  },

  /**
   * Module 2: Feature & Edge Analysis
   */
  async analyzeFeatures(formData) {
    return this._postMultipart(`${API_BASE}/image/features`, formData);
  },

  /**
   * Module 3: Image Segmentation
   */
  async segmentImage(formData) {
    return this._postMultipart(`${API_BASE}/image/segment`, formData);
  },

  /**
   * Module 4: YOLOv8 Object Detection
   */
  async detectObjects(formData) {
    return this._postMultipart(`${API_BASE}/image/detect`, formData);
  },

  /**
   * Module 5: Video Telemetry & Keyframes
   */
  async getVideoInfo(formData) {
    return this._postMultipart(`${API_BASE}/video/info`, formData);
  },

  /**
   * Module 6: YOLOv8 + ByteTrack Object Tracking
   */
  async trackVideo(formData) {
    return this._postMultipart(`${API_BASE}/video/track`, formData);
  },

  /**
   * Module 7: Motion & Flow Analysis
   */
  async analyzeMotion(formData) {
    return this._postMultipart(`${API_BASE}/video/motion`, formData);
  },

  /**
   * Internal multipart POST helper with error handling.
   */
  async _postMultipart(url, formData) {
    try {
      const res = await fetch(url, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`API Error (${res.status}): ${errorText}`);
      }
      return await res.json();
    } catch (err) {
      console.error(`[API Error] ${url}:`, err);
      throw err;
    }
  },
};
