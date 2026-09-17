/**
 * app.js — Client Single Page Application Router and Lifecycle Coordinator.
 */
function navigate(mode, subModule = null) {
  // Update sidebar active states
  document.querySelectorAll(".nav-item").forEach(item => item.classList.remove("active"));
  const targetNav = document.getElementById(`nav-${mode}`);
  if (targetNav) targetNav.classList.add("active");

  // Close mobile sidebar if open
  const sidebar = document.getElementById("sidebar");
  if (sidebar && sidebar.classList.contains("open")) {
    sidebar.classList.remove("open");
  }

  // Route to corresponding page view
  if (mode === "home") {
    HomePage.render();
  } else if (mode === "image") {
    ImageModePage.render(subModule || "preprocessing");
  } else if (mode === "video") {
    VideoModePage.render(subModule || "info");
  }
}

function toggleSidebar() {
  const sidebar = document.getElementById("sidebar");
  if (sidebar) {
    sidebar.classList.toggle("open");
  }
}

// Initialize application on DOM load
window.addEventListener("DOMContentLoaded", () => {
  const hash = window.location.hash.replace("#", "") || "home";
  navigate(hash);
});
