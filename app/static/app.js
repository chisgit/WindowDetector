// -------------------------------------------------------------
// Interactive State Variables
// -------------------------------------------------------------
let activeProjectId = null;
let projectMetadata = null;
let activePageNum = null;
let activePageObj = null;

// Drawing & Canvas state
const bgImage = new Image();
let windows = []; // Local window objects: [{ label: "W-01", box_px: [ymin, xmin, ymax, xmax] }]

// Mouse interaction states
let currentMode = "SELECT"; // "SELECT" or "ADD"
let activeBoxIndex = null;  // Index of the selected window box
let dragAction = null;      // "MOVE", "RESIZE_NW", "RESIZE_NE", "RESIZE_SW", "RESIZE_SE", or null
let dragStartPos = { x: 0, y: 0 }; // Starting offset for dragging a box
let modalEditIndex = null;  // Index of the box currently being edited in the modal

// Zoom & Pan state
let zoom = 1;
let panX = 0;
let panY = 0;
let isPanning = false;
let panStart = null;
let spaceDown = false;
let lastMouseDownTime = 0;

// Canvas Context
let canvas = null;
let ctx = null;

// -------------------------------------------------------------
// DOM Elements Initialization
// -------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  console.log("[DEBUG] DOM loaded. Initializing DOM element bindings.");
  
  canvas = document.getElementById("floorplan-canvas");
  ctx = canvas.getContext("2d");
  
  // Header Elements
  const pdfInput = document.getElementById("pdf-file-input");
  const uploadTriggerBtn = document.getElementById("upload-trigger-btn");
  const projectDropdown = document.getElementById("project-dropdown");
  const deleteProjectBtn = document.getElementById("delete-project-btn");
  
  // Sidebar Elements
  const thumbnailsList = document.getElementById("thumbnails-list");
  
  // Canvas Viewport Elements
  const detectBtn = document.getElementById("detect-btn");
  const addModeBtn = document.getElementById("add-mode-btn");
  const fitBtn = document.getElementById("fit-btn");
  const saveBtn = document.getElementById("save-btn");
  const canvasPlaceholder = document.getElementById("canvas-placeholder");
  const canvasWrapper = document.getElementById("canvas-wrapper");
  const canvasViewport = document.getElementById("canvas-viewport");
  
  // Loading Overlay
  const loadingOverlay = document.getElementById("loading-overlay");
  const loaderTitle = document.getElementById("loader-title");
  const loaderDesc = document.getElementById("loader-desc");
  
  // Status Bar
  const statusText = document.getElementById("status-text");
  const statusDot = document.getElementById("status-dot");
  
  // Modal Elements
  const editModal = document.getElementById("edit-modal");
  const modalCloseX = document.getElementById("modal-close-x");
  const modalCancelBtn = document.getElementById("modal-cancel-btn");
  const modalDeleteBtn = document.getElementById("modal-delete-btn");
  const modalSaveBtn = document.getElementById("modal-save-btn");
  const windowLabelInput = document.getElementById("window-label-input");
  const confirmDeleteModal = document.getElementById("confirm-delete-modal");
  const confirmDeleteDesc = document.getElementById("confirm-delete-desc");
  const confirmDeleteCancelBtn = document.getElementById("confirm-delete-cancel-btn");
  const confirmDeleteOkBtn = document.getElementById("confirm-delete-ok-btn");

  // -----------------------------------------------------------
  // Helper Functions: UI Notifications & Loader Control
  // -----------------------------------------------------------
  function updateStatus(message, isWorking = false, isError = false) {
    statusText.textContent = message;
    statusDot.className = "status-dot";
    
    if (isWorking) {
      statusDot.classList.add("yellow");
    } else if (isError) {
      statusDot.classList.add("red"); // fallback class
      statusDot.style.background = "#ef4444";
      statusDot.style.boxShadow = "0 0 8px #ef4444";
    } else {
      statusDot.classList.add("green");
      statusDot.style.background = "";
      statusDot.style.boxShadow = "";
    }
    console.log(`[DEBUG] Status: ${message}`);
  }

  function showLoader(title, description) {
    loaderTitle.textContent = title;
    loaderDesc.textContent = description;
    loadingOverlay.classList.remove("hidden");
  }

  function hideLoader() {
    loadingOverlay.classList.add("hidden");
  }

  // -----------------------------------------------------------
  // Zoom / Pan Helpers
  // -----------------------------------------------------------
  function applyTransform() {
    canvasViewport.style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`;
    console.log(`[DEBUG] applyTransform: zoom=${zoom.toFixed(3)}, panX=${Math.round(panX)}, panY=${Math.round(panY)}`);
  }

  function fitToView() {
    if (!bgImage.naturalWidth) return;
    const wrapW = canvasWrapper.clientWidth;
    const wrapH = canvasWrapper.clientHeight;
    const padding = 40;
    const newZoom = Math.min(
      (wrapW - padding * 2) / bgImage.naturalWidth,
      (wrapH - padding * 2) / bgImage.naturalHeight
    );
    zoom = newZoom;
    panX = (wrapW - bgImage.naturalWidth * zoom) / 2;
    panY = (wrapH - bgImage.naturalHeight * zoom) / 2;
    applyTransform();
    console.log(`[DEBUG] fitToView: fitted canvas at zoom=${zoom.toFixed(3)}`);
  }

  // -----------------------------------------------------------
  // Canvas Coordinate Conversion Formula
  // Translates a browser MouseEvent into original image pixel coords,
  // accounting for current zoom level and pan offset.
  // -----------------------------------------------------------
  function getMousePosOnImage(evt) {
    const wrapperRect = canvasWrapper.getBoundingClientRect();

    // Position of mouse relative to the wrapper div
    const localX = evt.clientX - wrapperRect.left;
    const localY = evt.clientY - wrapperRect.top;

    // Undo the pan/zoom transform to get true image-pixel coordinates
    const imageX = Math.round((localX - panX) / zoom);
    const imageY = Math.round((localY - panY) / zoom);

    return { x: imageX, y: imageY };
  }

  // -----------------------------------------------------------
  // Collision Detection Functions (Original pixel scale)
  // -----------------------------------------------------------
  function getBoxIndexAtPosition(mouseX, mouseY) {
    // Traverse backwards so the top-most overlay is selected first
    for (let i = windows.length - 1; i >= 0; i--) {
      const [ymin, xmin, ymax, xmax] = windows[i].box_px;
      if (mouseX >= xmin && mouseX <= xmax && mouseY >= ymin && mouseY <= ymax) {
        return i;
      }
    }
    return null;
  }

  function getHandleAtPosition(win, mouseX, mouseY) {
    const [ymin, xmin, ymax, xmax] = win.box_px;
    // Handle width is 8x8 pixels. Tolerance defines hit region radius.
    const tolerance = 15; 
    
    if (Math.abs(mouseX - xmin) < tolerance && Math.abs(mouseY - ymin) < tolerance) return "NW";
    if (Math.abs(mouseX - xmax) < tolerance && Math.abs(mouseY - ymin) < tolerance) return "NE";
    if (Math.abs(mouseX - xmin) < tolerance && Math.abs(mouseY - ymax) < tolerance) return "SW";
    if (Math.abs(mouseX - xmax) < tolerance && Math.abs(mouseY - ymax) < tolerance) return "SE";
    
    return null;
  }

  // -----------------------------------------------------------
  // Canvas Redraw Logic
  // -----------------------------------------------------------
  function drawCanvas() {
    if (!bgImage.src) {
      canvasPlaceholder.style.display = "flex";
      return;
    }
    
    canvasPlaceholder.style.display = "none";
    
    // Clear canvas viewport
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // 1. Draw floor plan image as backing layer
    ctx.drawImage(bgImage, 0, 0);
    
    // 2. Iterate and draw each window box overlay
    windows.forEach((win, index) => {
      const [ymin, xmin, ymax, xmax] = win.box_px;
      const width = xmax - xmin;
      const height = ymax - ymin;
      
      // Draw filled translucent rectangle (0.4 alpha emerald)
      ctx.fillStyle = "rgba(16, 185, 129, 0.4)";
      ctx.fillRect(xmin, ymin, width, height);
      
      // Draw outlines (thicker if currently selected)
      const isActive = (index === activeBoxIndex);
      ctx.strokeStyle = isActive ? "#10b981" : "rgba(16, 185, 129, 0.7)";
      ctx.lineWidth = isActive ? 4 : 2;
      ctx.strokeRect(xmin, ymin, width, height);
      
      // 3. Draw Label Pill above the bounding box
      ctx.font = "bold 14px Outfit";
      const textWidth = ctx.measureText(win.label).width;
      
      // Pill box background (solid dark container so label is readable)
      ctx.fillStyle = "rgba(10, 10, 12, 0.85)";
      ctx.fillRect(xmin, ymin - 25, textWidth + 12, 20);
      
      // Text
      ctx.fillStyle = "#ffffff";
      ctx.fillText(win.label, xmin + 6, ymin - 10);
      
      // 4. Draw Resizing Handles for selected box
      if (isActive) {
        ctx.fillStyle = "#ffffff";
        ctx.strokeStyle = "#0a0a0c";
        ctx.lineWidth = 1.5;
        
        const corners = [
          [xmin, ymin], // NW
          [xmax, ymin], // NE
          [xmin, ymax], // SW
          [xmax, ymax]  // SE
        ];
        
        corners.forEach(([cx, cy]) => {
          ctx.fillRect(cx - 5, cy - 5, 10, 10);
          ctx.strokeRect(cx - 5, cy - 5, 10, 10);
        });
      }
    });
    
    console.log(`[DEBUG] drawCanvas: Re-drawn layout with ${windows.length} window boundaries.`);
  }

  // -----------------------------------------------------------
  // Wheel Zoom (scroll on wrapper to zoom in/out around cursor)
  // -----------------------------------------------------------
  canvasWrapper.addEventListener("wheel", (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    const wrapperRect = canvasWrapper.getBoundingClientRect();
    const mx = e.clientX - wrapperRect.left;
    const my = e.clientY - wrapperRect.top;

    // Zoom toward the cursor position
    const ix = (mx - panX) / zoom;
    const iy = (my - panY) / zoom;
    zoom = Math.max(0.05, Math.min(20, zoom * factor));
    panX = mx - ix * zoom;
    panY = my - iy * zoom;
    applyTransform();
  }, { passive: false });

  // -----------------------------------------------------------
  // Space + Drag Panning
  // -----------------------------------------------------------
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !e.target.matches("input, select, textarea")) {
      spaceDown = true;
      canvasWrapper.style.cursor = "grab";
      e.preventDefault();
    }
  });

  window.addEventListener("keyup", (e) => {
    if (e.code === "Space") {
      spaceDown = false;
      canvasWrapper.style.cursor = "";
    }
  });

  canvasWrapper.addEventListener("mousedown", (e) => {
    const now = Date.now();
    const isRapidSecondClick = (now - lastMouseDownTime) < 300;
    lastMouseDownTime = now;

    if (spaceDown) {
      isPanning = true;
      panStart = { x: e.clientX - panX, y: e.clientY - panY };
      canvasWrapper.style.cursor = "grabbing";
      e.preventDefault();
      return;
    }

    // Double-click + drag on empty canvas space to pan
    if (isRapidSecondClick && bgImage.src) {
      const mousePos = getMousePosOnImage(e);
      const boxIdx = getBoxIndexAtPosition(mousePos.x, mousePos.y);
      if (boxIdx === null) {
        isPanning = true;
        panStart = { x: e.clientX - panX, y: e.clientY - panY };
        canvasWrapper.style.cursor = "grabbing";
        e.preventDefault();
        return;
      }
    }
  });

  canvasWrapper.addEventListener("mousemove", (e) => {
    if (isPanning && panStart) {
      panX = e.clientX - panStart.x;
      panY = e.clientY - panStart.y;
      applyTransform();
    }
  });

  canvasWrapper.addEventListener("mouseup", () => {
    if (isPanning) {
      isPanning = false;
      panStart = null;
      canvasWrapper.style.cursor = spaceDown ? "grab" : "";
    }
  });

  // -----------------------------------------------------------
  // Canvas Mouse Interactions State Machine
  // -----------------------------------------------------------
  canvas.addEventListener("mousedown", (e) => {
    if (isPanning || spaceDown) return; // let pan handler take over
    if (!bgImage.src) return;
    
    const mousePos = getMousePosOnImage(e);
    const mouseX = mousePos.x;
    const mouseY = mousePos.y;
    
    console.log(`[DEBUG] Canvas MouseDown: original pixel location (${mouseX}, ${mouseY})`);
    
    if (currentMode === "ADD") {
      console.log("[DEBUG] Canvas MouseDown: Adding new window starting at coordinate.");
      // Create empty window
      const newLabel = `W-${String(windows.length + 1).padStart(2, "0")}`;
      windows.push({
        label: newLabel,
        box_px: [mouseY, mouseX, mouseY + 2, mouseX + 2]
      });
      activeBoxIndex = windows.length - 1;
      dragAction = "RESIZE_SE"; // Immediately resize bottom right corner
      
      // Disable add mode trigger once started
      currentMode = "SELECT";
      addModeBtn.classList.remove("active");
      addModeBtn.textContent = "+ Add Window Mode: Off";
      
      drawCanvas();
      return;
    }
    
    if (currentMode === "SELECT") {
      // 1. Check handles of currently selected box first
      if (activeBoxIndex !== null) {
        const handle = getHandleAtPosition(windows[activeBoxIndex], mouseX, mouseY);
        if (handle) {
          dragAction = `RESIZE_${handle}`;
          console.log(`[DEBUG] Canvas MouseDown: Initializing handle drag: ${dragAction}`);
          return;
        }
      }
      
      // 2. Check if clicked inside another box
      const boxIdx = getBoxIndexAtPosition(mouseX, mouseY);
      if (boxIdx !== null) {
        activeBoxIndex = boxIdx;
        dragAction = "MOVE";
        
        // Save initial offset inside the box
        const clickedBox = windows[boxIdx].box_px;
        dragStartPos = {
          x: mouseX - clickedBox[1], // mouseX - xmin
          y: mouseY - clickedBox[0]  // mouseY - ymin
        };
        console.log(`[DEBUG] Canvas MouseDown: Grabbed Box ${boxIdx} for MOVE action. Offset: x=${dragStartPos.x}, y=${dragStartPos.y}`);
      } else {
        // Clicked empty canvas space
        activeBoxIndex = null;
        console.log("[DEBUG] Canvas MouseDown: Clicked background. Deselecting active overlay.");
      }
      
      drawCanvas();
    }
  });

  canvas.addEventListener("mousemove", (e) => {
    if (!bgImage.src) return;
    
    const mousePos = getMousePosOnImage(e);
    const mouseX = mousePos.x;
    const mouseY = mousePos.y;
    
    // Cursor mapping (hover guidelines)
    if (!dragAction) {
      if (activeBoxIndex !== null) {
        const handle = getHandleAtPosition(windows[activeBoxIndex], mouseX, mouseY);
        if (handle === "NW" || handle === "SE") {
          canvas.style.cursor = "nwse-resize";
          return;
        } else if (handle === "NE" || handle === "SW") {
          canvas.style.cursor = "nesw-resize";
          return;
        }
      }
      
      const hoverBox = getBoxIndexAtPosition(mouseX, mouseY);
      if (hoverBox !== null) {
        canvas.style.cursor = "move";
      } else {
        canvas.style.cursor = (currentMode === "ADD") ? "crosshair" : "default";
      }
      return;
    }
    
    // Dragging Actions
    const box = windows[activeBoxIndex].box_px;
    
    if (dragAction === "MOVE") {
      const height = box[2] - box[0];
      const width = box[3] - box[1];
      
      // Calculate target location with boundary constraints
      let newXmin = Math.max(0, Math.min(mouseX - dragStartPos.x, bgImage.naturalWidth - width));
      let newYmin = Math.max(0, Math.min(mouseY - dragStartPos.y, bgImage.naturalHeight - height));
      
      windows[activeBoxIndex].box_px = [
        newYmin,
        newXmin,
        newYmin + height,
        newXmin + width
      ];
    } 
    else if (dragAction.startsWith("RESIZE_")) {
      const handle = dragAction.replace("RESIZE_", "");
      
      if (handle === "SE") {
        box[2] = Math.max(0, Math.min(mouseY, bgImage.naturalHeight)); // ymax
        box[3] = Math.max(0, Math.min(mouseX, bgImage.naturalWidth));  // xmax
      } else if (handle === "NW") {
        box[0] = Math.max(0, Math.min(mouseY, bgImage.naturalHeight)); // ymin
        box[1] = Math.max(0, Math.min(mouseX, bgImage.naturalWidth));  // xmin
      } else if (handle === "NE") {
        box[0] = Math.max(0, Math.min(mouseY, bgImage.naturalHeight)); // ymin
        box[3] = Math.max(0, Math.min(mouseX, bgImage.naturalWidth));  // xmax
      } else if (handle === "SW") {
        box[2] = Math.max(0, Math.min(mouseY, bgImage.naturalHeight)); // ymax
        box[1] = Math.max(0, Math.min(mouseX, bgImage.naturalWidth));  // xmin
      }
    }
    
    drawCanvas();
  });

  canvas.addEventListener("mouseup", () => {
    if (!bgImage.src || !dragAction) return;
    
    console.log(`[DEBUG] Canvas MouseUp: Finalizing drag operation '${dragAction}'`);
    dragAction = null;
    
    // Normalization: Ensure coordinates remain positive (swap boundaries if dragged negative direction)
    const box = windows[activeBoxIndex].box_px;
    if (box[0] > box[2]) {
      const temp = box[0];
      box[0] = box[2];
      box[2] = temp;
    }
    if (box[1] > box[3]) {
      const temp = box[1];
      box[1] = box[3];
      box[3] = temp;
    }
    
    // If the window is too small (e.g. user clicked by accident), filter/delete it
    const width = box[3] - box[1];
    const height = box[2] - box[0];
    if (width < 5 || height < 5) {
      console.log("[DEBUG] Canvas MouseUp: Bounding box too small, discarding shape.");
      windows.splice(activeBoxIndex, 1);
      activeBoxIndex = null;
    }
    
    drawCanvas();
  });

  // Open Edit Modal on Double Click
  canvas.addEventListener("dblclick", (e) => {
    if (!bgImage.src) return;
    
    const mousePos = getMousePosOnImage(e);
    const boxIdx = getBoxIndexAtPosition(mousePos.x, mousePos.y);
    
    if (boxIdx !== null) {
      console.log(`[DEBUG] Canvas DoubleClick: Open modal for Box Index ${boxIdx}`);
      modalEditIndex = boxIdx;
      windowLabelInput.value = windows[boxIdx].label;
      editModal.classList.remove("hidden");
      windowLabelInput.focus();
    }
  });

  // -----------------------------------------------------------
  // Modal Buttons Action Event Listeners
  // -----------------------------------------------------------
  modalCloseX.addEventListener("click", () => { editModal.classList.add("hidden"); });
  modalCancelBtn.addEventListener("click", () => { editModal.classList.add("hidden"); });
  
  modalSaveBtn.addEventListener("click", () => {
    if (modalEditIndex !== null) {
      const updatedLabel = windowLabelInput.value.trim();
      windows[modalEditIndex].label = updatedLabel ? updatedLabel : `W-${modalEditIndex+1}`;
      console.log(`[DEBUG] Modal Save: Renamed window ${modalEditIndex} to '${windows[modalEditIndex].label}'`);
      editModal.classList.add("hidden");
      drawCanvas();
    }
  });
  
  modalDeleteBtn.addEventListener("click", () => {
    if (modalEditIndex !== null) {
      console.log(`[DEBUG] Modal Delete: Removing window box index ${modalEditIndex}`);
      windows.splice(modalEditIndex, 1);
      modalEditIndex = null;
      activeBoxIndex = null;
      editModal.classList.add("hidden");
      drawCanvas();
    }
  });

  // -----------------------------------------------------------
  // API and Sidebar Selection Integration
  // -----------------------------------------------------------
  
  // 1. PDF Upload Action Trigger
  uploadTriggerBtn.addEventListener("click", () => {
    pdfInput.click();
  });
  
  pdfInput.addEventListener("change", () => {
    if (pdfInput.files.length === 0) return;
    
    const file = pdfInput.files[0];
    console.log(`[DEBUG] PDF Upload initiated for: ${file.name}`);
    
    const formData = new FormData();
    formData.append("file", file);
    
    showLoader("Uploading PDF Blueprint", "Extracting images and organizing project files...");
    updateStatus("Uploading blueprint document...", true);
    
    fetch("/api/upload", {
      method: "POST",
      body: formData
    })
    .then(res => {
      if (!res.ok) throw new Error("Upload processing error on server.");
      return res.json();
    })
    .then(data => {
      console.log("[DEBUG] PDF Upload succeeded. Response:", data);
      hideLoader();
      updateStatus("Upload completed successfully. Loading project blueprints.");
      
      // Update projects list dropdown and select new project
      loadProjects(data.project_id);
    })
    .catch(err => {
      console.error("[ERROR] PDF Upload failed:", err);
      hideLoader();
      updateStatus("Upload failed. Please ensure file is a valid PDF.", false, true);
      alert("Error uploading PDF file. Please try again.");
    });
  });

  // 2. Select Project Dropdown Action
  projectDropdown.addEventListener("change", () => {
    const selId = projectDropdown.value;
    if (!selId) {
      resetAppState();
      return;
    }
    activeProjectId = selId;
    deleteProjectBtn.removeAttribute("disabled");
    loadProjectMetadata(selId);
  });

  deleteProjectBtn.addEventListener("click", () => {
    if (!activeProjectId || !projectMetadata) return;
    confirmDeleteDesc.textContent = `Permanently delete "${projectMetadata.pdf_name}" and all rendered pages, thumbnails, and detections? This cannot be undone.`;
    confirmDeleteModal.classList.remove("hidden");
  });

  confirmDeleteCancelBtn.addEventListener("click", () => {
    confirmDeleteModal.classList.add("hidden");
  });

  confirmDeleteOkBtn.addEventListener("click", () => {
    if (!activeProjectId) return;

    updateStatus("Deleting project...", true);
    confirmDeleteOkBtn.setAttribute("disabled", "true");

    fetch(`/api/projects/${activeProjectId}`, { method: "DELETE" })
      .then(res => {
        if (!res.ok) {
          return res.json().then(errData => {
            throw new Error(errData.detail || "Delete endpoint failed.");
          });
        }
        return res.json();
      })
      .then(() => {
        confirmDeleteModal.classList.add("hidden");
        confirmDeleteOkBtn.removeAttribute("disabled");
        resetAppState();
        loadProjects();
        updateStatus("Project deleted.");
      })
      .catch(err => {
        console.error("[ERROR] Failed to delete project:", err);
        confirmDeleteOkBtn.removeAttribute("disabled");
        updateStatus(`Delete failed: ${err.message}`, false, true);
        alert(`Delete failed: ${err.message}`);
      });
  });

  // Fetch projects list
  function loadProjects(selectId = null) {
    updateStatus("Fetching projects list...", true);
    fetch("/api/projects")
      .then(res => res.json())
      .then(data => {
        projectDropdown.innerHTML = '<option value="">-- Select Active Project --</option>';
        data.forEach(p => {
          const opt = document.createElement("option");
          opt.value = p.project_id;
          opt.textContent = `${p.pdf_name} (${p.floorplan_count}/${p.total_pages} Floorplans)`;
          projectDropdown.appendChild(opt);
        });
        
        if (selectId) {
          projectDropdown.value = selectId;
          activeProjectId = selectId;
          deleteProjectBtn.removeAttribute("disabled");
          loadProjectMetadata(selectId);
        } else {
          deleteProjectBtn.setAttribute("disabled", "true");
          updateStatus("Projects list refreshed.");
        }
      })
      .catch(err => {
        console.error("[ERROR] Failed to fetch projects list:", err);
        updateStatus("Failed to retrieve projects list.", false, true);
      });
  }

  // Fetch Project Metadata & Build Sidebar Thumbnails
  function loadProjectMetadata(projId) {
    updateStatus("Loading project blueprints...", true);
    fetch(`/api/projects/${projId}/metadata`)
      .then(res => res.json())
      .then(meta => {
        projectMetadata = meta;
        console.log("[DEBUG] Loaded project metadata:", meta);
        
        // Clear sidebar
        thumbnailsList.innerHTML = "";
        
        const validPages = meta.pages || [];
        if (validPages.length === 0) {
          thumbnailsList.innerHTML = '<div class="no-pages-message">No pages found.</div>';
          updateStatus("Project is empty.");
          return;
        }
        
        validPages.forEach(p => {
          const card = document.createElement("div");
          card.className = "thumbnail-card";
          
          if (!p.is_floor_plan) {
            card.classList.add("filtered-out");
          }
          if (p.user_corrected) {
            card.classList.add("corrected");
          }
          
          // Image wrapper
          const imgWrapper = document.createElement("div");
          imgWrapper.className = "thumbnail-img-wrapper";
          
          const img = document.createElement("img");
          // Fetch low-res thumbnail endpoint to protect browser memory
          img.src = `/api/projects/${projId}/pages/${p.page_number}/thumbnail`;
          img.alt = `Page ${p.page_number}`;
          imgWrapper.appendChild(img);
          card.appendChild(imgWrapper);
          
          // Info row
          const info = document.createElement("div");
          info.className = "thumbnail-info";
          
          const num = document.createElement("span");
          num.className = "thumbnail-number";
          num.textContent = `Page ${p.page_number}`;
          info.appendChild(num);
          
          const badge = document.createElement("span");
          badge.className = "thumbnail-badge";
          if (p.user_corrected) {
            badge.textContent = "Reviewed";
          } else if (p.is_floor_plan) {
            badge.textContent = "Floorplan";
          } else {
            badge.textContent = "Text / Spec";
          }
          info.appendChild(badge);
          card.appendChild(info);
          
          // Click event: Select page
          card.addEventListener("click", () => {
            // Remove active classes
            document.querySelectorAll(".thumbnail-card").forEach(c => c.classList.remove("active"));
            card.classList.add("active");
            selectPage(p);
          });
          
          thumbnailsList.appendChild(card);
        });
        
        // Auto select first page that matches floor plan
        const firstFloorplan = validPages.find(p => p.is_floor_plan);
        if (firstFloorplan) {
          const idx = validPages.indexOf(firstFloorplan);
          const cards = thumbnailsList.querySelectorAll(".thumbnail-card");
          if (cards[idx]) {
            cards[idx].classList.add("active");
            selectPage(firstFloorplan);
          }
        } else if (validPages.length > 0) {
          // Fallback to page 1
          const cards = thumbnailsList.querySelectorAll(".thumbnail-card");
          if (cards[0]) {
            cards[0].classList.add("active");
            selectPage(validPages[0]);
          }
        }
        
        updateStatus("Blueprints loaded. Selected active page.");
      })
      .catch(err => {
        console.error("[ERROR] Failed to load project metadata:", err);
        updateStatus("Failed to load project details.", false, true);
      });
  }

  // Load target page on Canvas
  function selectPage(pageObj) {
    console.log(`[DEBUG] selectPage: Rendering Page ${pageObj.page_number} on Canvas.`);
    activePageNum = pageObj.page_number;
    activePageObj = pageObj;
    
    // Set controls states
    detectBtn.removeAttribute("disabled");
    addModeBtn.removeAttribute("disabled");
    fitBtn.removeAttribute("disabled");
    saveBtn.removeAttribute("disabled");
    
    // Reset canvas tracking states
    activeBoxIndex = null;
    currentMode = "SELECT";
    addModeBtn.classList.remove("active");
    addModeBtn.textContent = "+ Add Window Mode: Off";
    
    // Reset window boundaries array (copy coordinates to prevent mutated saves without clicks)
    windows = JSON.parse(JSON.stringify(pageObj.windows || []));
    
    updateStatus(`Loading blueprint canvas (Page ${activePageNum})...`, true);
    
    // Load high-resolution image to canvas background
    bgImage.onload = () => {
      canvas.width = bgImage.naturalWidth;
      canvas.height = bgImage.naturalHeight;
      console.log(`[DEBUG] selectPage: Image loaded successfully. Resolution: ${canvas.width}x${canvas.height}`);

      // Fit the image into the visible viewport automatically
      fitToView();
      drawCanvas();
      updateStatus(`Blueprint loaded. Detected windows count: ${windows.length}`);
    };
    
    bgImage.onerror = () => {
      console.error(`[ERROR] selectPage: Failed to load background image: ${bgImage.src}`);
      updateStatus("Failed to render background blueprint image.", false, true);
    };
    
    bgImage.src = `/api/projects/${activeProjectId}/pages/${activePageNum}/image`;
  }

  // Reset App states when no project selected
  function resetAppState() {
    activeProjectId = null;
    projectMetadata = null;
    activePageNum = null;
    activePageObj = null;
    windows = [];
    activeBoxIndex = null;
    bgImage.src = "";
    
    detectBtn.setAttribute("disabled", "true");
    addModeBtn.setAttribute("disabled", "true");
    fitBtn.setAttribute("disabled", "true");
    saveBtn.setAttribute("disabled", "true");
    deleteProjectBtn.setAttribute("disabled", "true");
    
    thumbnailsList.innerHTML = '<div class="no-pages-message">No PDF uploaded yet.</div>';
    canvasPlaceholder.style.display = "flex";
    updateStatus("System Reset. Please select or upload a project.");
  }

  // -----------------------------------------------------------
  // Action Buttons Listeners
  // -----------------------------------------------------------

  // Fit Page Button
  fitBtn.addEventListener("click", () => {
    fitToView();
    updateStatus("View fitted to page.");
  });

  // Toggle ADD Window Mode
  addModeBtn.addEventListener("click", () => {
    if (currentMode === "SELECT") {
      currentMode = "ADD";
      addModeBtn.classList.add("active");
      addModeBtn.textContent = "+ Add Window Mode: On";
      activeBoxIndex = null;
      drawCanvas();
      updateStatus("Add Mode ON: Click and drag on empty canvas to draw a window box.");
    } else {
      currentMode = "SELECT";
      addModeBtn.classList.remove("active");
      addModeBtn.textContent = "+ Add Window Mode: Off";
      updateStatus("Add Mode OFF: Normal coordinate editor active.");
    }
  });

  // Run AI Detection
  detectBtn.addEventListener("click", () => {
    if (!activeProjectId || !activePageNum) return;
    
    showLoader("Analyzing Floor Plan", "Asking Gemini to identify window layouts based on blueprint lines...");
    updateStatus("Running AI window detection query...", true);
    
    fetch(`/api/projects/${activeProjectId}/pages/${activePageNum}/detect`)
      .then(res => {
        if (!res.ok) {
          // Read server JSON error detail
          return res.json().then(errData => {
            throw new Error(errData.detail || "API detection error.");
          });
        }
        return res.json();
      })
      .then(data => {
        console.log("[DEBUG] AI Detection succeeded. Response:", data);
        windows = data.windows || [];
        activeBoxIndex = null;
        drawCanvas();
        hideLoader();
        updateStatus(`Gemini window detection complete. Detected: ${windows.length} windows.`);
      })
      .catch(err => {
        console.error("[ERROR] Window detection failed:", err);
        hideLoader();
        updateStatus(`Detection failed: ${err.message}`, false, true);
        alert(`AI Detection failed: ${err.message}`);
      });
  });

  // Save Corrections
  saveBtn.addEventListener("click", () => {
    if (!activeProjectId || !activePageNum) return;
    
    updateStatus("Saving window boundaries metadata to project storage...", true);
    
    const payload = {
      windows: windows.map(w => ({
        label: w.label,
        box_px: w.box_px
      }))
    };
    
    fetch(`/api/projects/${activeProjectId}/pages/${activePageNum}/save`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    })
    .then(res => {
      if (!res.ok) throw new Error("Save endpoint failed on server.");
      return res.json();
    })
    .then(data => {
      console.log("[DEBUG] Save adjustments succeeded. Response:", data);
      updateStatus("Corrections saved successfully. Blueprint metadata updated.");
      
      // Update cached page metadata object locally
      if (activePageObj) {
        activePageObj.windows = JSON.parse(JSON.stringify(windows));
        activePageObj.user_corrected = true;
      }
      
      // Refresh sidebar list to update correct badges/status without reloading the dropdown
      const activeCard = thumbnailsList.querySelector(".thumbnail-card.active");
      if (activeCard) {
        activeCard.classList.add("corrected");
        const badge = activeCard.querySelector(".thumbnail-badge");
        if (badge) badge.textContent = "Reviewed";
      }
    })
    .catch(err => {
      console.error("[ERROR] Failed to save corrections:", err);
      updateStatus("Failed to write coordinates to local file.", false, true);
      alert("Error saving corrections. Check network connection.");
    });
  });

  // Initial load
  loadProjects();
});
