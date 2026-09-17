/**
 * dashboard.js — Dynamic Chart.js 4.4 Visualizations Component.
 * Renders Histograms, Donut Category Breakdowns, Polar Direction Charts, and Motion Energy Curves.
 */
const Dashboard = {
  _charts: {},

  /**
   * Render or update a 256-bin Intensity Histogram.
   */
  renderHistogram(canvasId, histData) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (this._charts[canvasId]) {
      this._charts[canvasId].destroy();
    }

    const labels = Array.from({ length: 256 }, (_, i) => i);
    const datasets = [];

    if (histData.original) {
      datasets.push({
        label: "Original Intensity",
        data: histData.original,
        borderColor: "rgba(148, 163, 184, 0.8)",
        backgroundColor: "rgba(148, 163, 184, 0.15)",
        borderWidth: 1.5,
        fill: true,
        pointRadius: 0,
      });
    }

    if (histData.equalized) {
      datasets.push({
        label: "Equalized / Processed",
        data: histData.equalized,
        borderColor: "rgba(6, 182, 212, 1)",
        backgroundColor: "rgba(6, 182, 212, 0.25)",
        borderWidth: 1.5,
        fill: true,
        pointRadius: 0,
      });
    }

    this._charts[canvasId] = new Chart(ctx, {
      type: "line",
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: "#94a3b8", font: { family: "Inter", size: 11 } } },
        },
        scales: {
          x: {
            grid: { color: "rgba(255, 255, 255, 0.04)" },
            ticks: { color: "#64748b", maxTicksLimit: 8 },
          },
          y: {
            grid: { color: "rgba(255, 255, 255, 0.04)" },
            ticks: { color: "#64748b" },
          },
        },
      },
    });
  },

  /**
   * Render Category Breakdown Donut Chart for YOLO detections.
   */
  renderCategoryDonut(canvasId, categoryCounts) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (this._charts[canvasId]) {
      this._charts[canvasId].destroy();
    }

    const labels = Object.keys(categoryCounts);
    const data = Object.values(categoryCounts);

    const colors = [
      "#06b6d4", "#6366f1", "#10b981", "#f59e0b",
      "#ef4444", "#a855f7", "#ec4899", "#14b8a6"
    ];

    this._charts[canvasId] = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels,
        datasets: [{
          data,
          backgroundColor: colors.slice(0, labels.length),
          borderColor: "#0d1322",
          borderWidth: 2,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "right",
            labels: { color: "#94a3b8", font: { family: "Inter", size: 12 } },
          },
        },
      },
    });
  },

  /**
   * Render 8-Direction Polar Area Chart for Optical Flow.
   */
  renderPolarFlow(canvasId, polarDistribution) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (this._charts[canvasId]) {
      this._charts[canvasId].destroy();
    }

    const labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];

    this._charts[canvasId] = new Chart(ctx, {
      type: "polarArea",
      data: {
        labels,
        datasets: [{
          data: polarDistribution,
          backgroundColor: [
            "rgba(6, 182, 212, 0.6)",
            "rgba(99, 102, 241, 0.6)",
            "rgba(16, 185, 129, 0.6)",
            "rgba(245, 158, 11, 0.6)",
            "rgba(239, 68, 68, 0.6)",
            "rgba(168, 85, 247, 0.6)",
            "rgba(236, 72, 153, 0.6)",
            "rgba(20, 184, 166, 0.6)",
          ],
          borderColor: "rgba(255, 255, 255, 0.1)",
          borderWidth: 1,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "right",
            labels: { color: "#94a3b8", font: { family: "Inter", size: 11 } },
          },
        },
        scales: {
          r: {
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#64748b", backdropColor: "transparent" },
          },
        },
      },
    });
  },
};
