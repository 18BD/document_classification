document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("file-input");
  const fileInputLabel = document.getElementById("file-input-label");
  const uploadFeedback = document.querySelector(".upload-feedback");
  const themeToggleBtn = document.getElementById("theme-toggle-btn");
  const htmlElement = document.documentElement;

  fileInputLabel.addEventListener("dragover", (e) => {
    e.preventDefault();
    fileInputLabel.classList.add("drag-over");
  });

  fileInputLabel.addEventListener("dragleave", () => {
    fileInputLabel.classList.remove("drag-over");
  });

  fileInputLabel.addEventListener("drop", (e) => {
    e.preventDefault();
    fileInputLabel.classList.remove("drag-over");
    const files = e.dataTransfer.files;
    handleFiles(files);
  });

  fileInput.addEventListener("change", () => {
    const files = fileInput.files;
    handleFiles(files);
  });

  function handleFiles(files) {
    uploadFeedback.textContent = "Files uploaded";
    uploadFeedback.classList.add("show");
  }

});
