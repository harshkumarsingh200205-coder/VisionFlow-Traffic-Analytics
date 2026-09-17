/**
 * progressOverlay.js — Global Loading & Processing Modal Controller.
 */
const ProgressOverlay = {
  show(title = "Processing…", sub = "Executing Computer Vision algorithms") {
    const el = document.getElementById("loading-overlay");
    const tEl = document.getElementById("loading-title");
    const sEl = document.getElementById("loading-sub");
    if (el) {
      if (tEl) tEl.innerText = title;
      if (sEl) sEl.innerText = sub;
      el.classList.remove("hidden");
    }
  },

  hide() {
    const el = document.getElementById("loading-overlay");
    if (el) {
      el.classList.add("hidden");
    }
  },
};
