import os
import uuid
import json
import shutil
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Import services
from app.services.pdf_processor import render_pdf_to_images, classify_pages_locally
try:
    from app.services.gemini_service import detect_windows, detect_windows_in_region
except Exception as _gemini_err:  # google-genai not installed: local deterministic model still works
    print(f"[WARN] Gemini service unavailable ({_gemini_err}); AI detection disabled, local deterministic model only.")
    def detect_windows(*a, **k):
        raise HTTPException(status_code=503, detail="Gemini AI detection not available (google-genai not installed).")
    def detect_windows_in_region(*a, **k):
        raise HTTPException(status_code=503, detail="Gemini AI detection not available (google-genai not installed).")
from app.services.plan_region_detector import detect_floor_plan_regions
from app.services.local_window_detector import detect_windows_locally
from app.services.stantec_detector_core import detect_stantec_plan_regions, detect_stantec_windows_in_region, is_stantec_pdf

# Load environment variables
load_dotenv()

app = FastAPI(title="Interactive Floor Plan Window Detector API")

# Configuration variables
PROJECTS_DIR = os.getenv("PROJECTS_DIR", "./projects")
PDF_RENDER_DPI = int(os.getenv("PDF_RENDER_DPI", "200"))

# Ensure projects directory exists
os.makedirs(PROJECTS_DIR, exist_ok=True)
print(f"[DEBUG] FastAPI initialization: Projects directory configured at '{PROJECTS_DIR}'")
print(f"[DEBUG] FastAPI initialization: Default PDF Render DPI set to {PDF_RENDER_DPI}")

# -------------------------------------------------------------
# Pydantic Schemas for Requests
# -------------------------------------------------------------
class WindowSaveItem(BaseModel):
    label: str
    box_px: list[int]  # [ymin_px, xmin_px, ymax_px, xmax_px]

class SaveWindowsRequest(BaseModel):
    windows: list[WindowSaveItem]

class DetectRegionRequest(BaseModel):
    region: list[int]  # [ymin_px, xmin_px, ymax_px, xmax_px]
    replace: bool = False
    backend: str = "legacy-local"


def append_correction_log(project_path: str, event: dict):
    """
    Stores each user-reviewed correction as JSONL. This creates a trackable
    learning trail without changing the page metadata format.
    """
    log_path = os.path.join(project_path, "correction_log.jsonl")
    event["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def _compute_box_iou(box_a: list[int], box_b: list[int]) -> float:
    ymin_a, xmin_a, ymax_a, xmax_a = box_a
    ymin_b, xmin_b, ymax_b, xmax_b = box_b

    inter_xmin = max(xmin_a, xmin_b)
    inter_ymin = max(ymin_a, ymin_b)
    inter_xmax = min(xmax_a, xmax_b)
    inter_ymax = min(ymax_a, ymax_b)

    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height

    area_a = max(0, xmax_a - xmin_a) * max(0, ymax_a - ymin_a)
    area_b = max(0, xmax_b - xmin_b) * max(0, ymax_b - ymin_b)

    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def _append_non_overlapping_windows(existing_windows: list[dict], detected_windows: list[dict], iou_threshold: float = 0.45) -> list[dict]:
    final_windows = existing_windows.copy()
    for candidate in detected_windows:
        overlaps = any(
            _compute_box_iou(candidate["box_px"], existing["box_px"]) >= iou_threshold
            for existing in existing_windows
        )
        if overlaps:
            continue
        final_windows.append(candidate)
    return final_windows

# -------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accepts PDF upload, generates a new project, extracts page images
    (high-res & thumbnail), filters for floor plans, and creates metadata.json.
    """
    print(f"\n[DEBUG] POST /api/upload: Received upload request for file '{file.filename}'")
    
    if not file.filename.lower().endswith('.pdf'):
        print("[ERROR] POST /api/upload: Uploaded file is not a PDF.")
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF document.")
        
    project_id = str(uuid.uuid4())
    project_path = os.path.join(PROJECTS_DIR, project_id)
    os.makedirs(project_path, exist_ok=True)
    print(f"[DEBUG] POST /api/upload: Created project directory at '{project_path}' (Project ID: {project_id})")
    
    # Save original PDF
    original_pdf_path = os.path.join(project_path, "original.pdf")
    try:
        with open(original_pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"[DEBUG] POST /api/upload: Saved original PDF successfully to '{original_pdf_path}'")
    except Exception as e:
        print(f"[ERROR] POST /api/upload: Failed to save PDF file. Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to write uploaded file to disk.")
        
    # Render PDF pages to images
    try:
        pages_metadata = render_pdf_to_images(original_pdf_path, project_path, dpi=PDF_RENDER_DPI)
    except Exception as e:
        print(f"[ERROR] POST /api/upload: PDF rendering failed. Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF rendering failed: {str(e)}")
        
    # Classify pages locally so upload does not depend on external AI quota.
    print(f"[DEBUG] POST /api/upload: Running local floor plan classification on {len(pages_metadata)} pages...")
    classified_pages = classify_pages_locally(pages_metadata)

    # Compile final pages list
    pages_list = []
    for page in classified_pages:
        is_floor = page.get("is_floor_plan", False)
        print(f"[DEBUG] POST /api/upload: Page {page['page_number']} classified as {'FLOOR PLAN' if is_floor else 'non-floor-plan'}")

        page_entry = {
            "page_number": page["page_number"],
            "image_name": page["image_name"],
            "thumbnail_name": page["thumbnail_name"],
            "is_floor_plan": is_floor,
            "width": page["width"],
            "height": page["height"],
            "plan_regions": [],
            "windows": [],
            "user_corrected": False
        }
        pages_list.append(page_entry)
        
    # Create final metadata dictionary
    metadata = {
        "project_id": project_id,
        "pdf_name": file.filename,
        "pages": pages_list
    }
    
    # Save metadata.json
    metadata_path = os.path.join(project_path, "metadata.json")
    try:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[DEBUG] POST /api/upload: Created metadata.json file at '{metadata_path}'")
    except Exception as e:
        print(f"[ERROR] POST /api/upload: Failed to write metadata.json. Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create project metadata.")
        
    return {"project_id": project_id, "pdf_name": file.filename, "total_pages": len(pages_list)}

@app.get("/api/projects")
async def list_projects():
    """
    Scans the projects folder and returns a summary list of all projects.
    """
    print("[DEBUG] GET /api/projects: Listing all projects...")
    projects = []
    
    if not os.path.exists(PROJECTS_DIR):
        return []
        
    for item in os.listdir(PROJECTS_DIR):
        item_path = os.path.join(PROJECTS_DIR, item)
        if os.path.isdir(item_path):
            meta_path = os.path.join(item_path, "metadata.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        meta_data = json.load(f)
                    
                    # Count pages and floor plans
                    all_pages = meta_data.get("pages", [])
                    floorplan_count = sum(1 for p in all_pages if p.get("is_floor_plan", False))
                    
                    projects.append({
                        "project_id": meta_data.get("project_id"),
                        "pdf_name": meta_data.get("pdf_name"),
                        "total_pages": len(all_pages),
                        "floorplan_count": floorplan_count
                    })
                except Exception as e:
                    print(f"[WARNING] GET /api/projects: Error reading metadata at '{meta_path}'. Error: {str(e)}")
                    
    print(f"[DEBUG] GET /api/projects: Found {len(projects)} valid projects.")
    return projects

@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    """
    Deletes a project directory and all derived files for a selected project.
    """
    print(f"\n[DEBUG] DELETE /api/projects/{project_id}: Delete requested")

    try:
        uuid.UUID(project_id)
    except ValueError:
        print("[ERROR] delete_project: Invalid project_id format.")
        raise HTTPException(status_code=400, detail="Invalid project id.")

    projects_root = os.path.abspath(PROJECTS_DIR)
    project_path = os.path.abspath(os.path.join(PROJECTS_DIR, project_id))

    if os.path.commonpath([projects_root, project_path]) != projects_root:
        print(f"[ERROR] delete_project: Refusing path outside projects dir: '{project_path}'")
        raise HTTPException(status_code=400, detail="Invalid project path.")

    if not os.path.isdir(project_path):
        print(f"[ERROR] delete_project: Project directory not found at '{project_path}'")
        raise HTTPException(status_code=404, detail="Project not found.")

    try:
        shutil.rmtree(project_path)
        print(f"[DEBUG] delete_project: Deleted project directory '{project_path}'")
    except Exception as e:
        print(f"[ERROR] delete_project: Failed to delete project. Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete project.")

    return {"status": "success", "project_id": project_id}

@app.get("/api/projects/{project_id}/metadata")
async def get_project_metadata(project_id: str):
    """
    Returns the complete metadata.json for the specified project.
    """
    print(f"[DEBUG] GET /api/projects/{project_id}/metadata: Fetching metadata")
    meta_path = os.path.join(PROJECTS_DIR, project_id, "metadata.json")
    
    if not os.path.exists(meta_path):
        print(f"[ERROR] GET /api/projects/{project_id}/metadata: Project metadata file not found at '{meta_path}'")
        raise HTTPException(status_code=404, detail="Project metadata not found.")
        
    try:
        with open(meta_path, "r") as f:
            meta_data = json.load(f)
        return meta_data
    except Exception as e:
        print(f"[ERROR] GET /api/projects/{project_id}/metadata: Failed to load JSON. Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to read project metadata.")

@app.get("/api/projects/{project_id}/pages/{page_num}/image")
async def get_page_image(project_id: str, page_num: int):
    """
    Returns the high-resolution PNG image for a given page.
    """
    print(f"[DEBUG] GET /api/projects/{project_id}/pages/{page_num}/image: Fetching page image")
    image_path = os.path.join(PROJECTS_DIR, project_id, f"page_{page_num}.png")
    
    if not os.path.exists(image_path):
        print(f"[ERROR] GET /api/projects/{project_id}/pages/{page_num}/image: Image not found at '{image_path}'")
        raise HTTPException(status_code=404, detail="Page image file not found.")
        
    return FileResponse(image_path)

@app.get("/api/projects/{project_id}/pages/{page_num}/thumbnail")
async def get_page_thumbnail(project_id: str, page_num: int):
    """
    Returns the low-resolution PNG thumbnail for a given page.
    """
    print(f"[DEBUG] GET /api/projects/{project_id}/pages/{page_num}/thumbnail: Fetching page thumbnail")
    thumb_path = os.path.join(PROJECTS_DIR, project_id, f"page_{page_num}_thumb.png")
    
    if not os.path.exists(thumb_path):
        print(f"[ERROR] GET /api/projects/{project_id}/pages/{page_num}/thumbnail: Thumbnail not found at '{thumb_path}'")
        # Fall back to main image if thumbnail is somehow missing
        main_img_path = os.path.join(PROJECTS_DIR, project_id, f"page_{page_num}.png")
        if os.path.exists(main_img_path):
            print(f"[WARNING] GET /api/projects/{project_id}/pages/{page_num}/thumbnail: Thumbnail missing, falling back to high-res image.")
            return FileResponse(main_img_path)
        raise HTTPException(status_code=404, detail="Page thumbnail file not found.")
        
    return FileResponse(thumb_path)

@app.get("/api/projects/{project_id}/pages/{page_num}/detect")
async def run_detection(project_id: str, page_num: int):
    """
    Triggers Gemini window detection on the high-res page image.
    Uses corrected metadata from previous pages as few-shot visual context.
    """
    print(f"\n[DEBUG] GET /api/projects/{project_id}/pages/{page_num}/detect: Triggering detection...")
    
    # 1. Safety Guard Check: Validate Gemini API Key presence
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.strip() == "":
        print("[ERROR] run_detection: GEMINI_API_KEY environment variable is empty or missing.")
        raise HTTPException(
            status_code=412,
            detail="Google API Key missing. Please configure GEMINI_API_KEY in your .env file."
        )
        
    # 2. Retrieve metadata
    project_path = os.path.join(PROJECTS_DIR, project_id)
    meta_path = os.path.join(project_path, "metadata.json")
    
    if not os.path.exists(meta_path):
        print(f"[ERROR] run_detection: Project metadata not found at '{meta_path}'")
        raise HTTPException(status_code=404, detail="Project metadata not found.")
        
    try:
        with open(meta_path, "r") as f:
            project_meta = json.load(f)
    except Exception as e:
        print(f"[ERROR] run_detection: Failed to load metadata.json. Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load project metadata.")
        
    # 3. Locate Target Page
    target_page = None
    pages = project_meta.get("pages", [])
    for page in pages:
        if page.get("page_number") == page_num:
            target_page = page
            break
            
    if not target_page:
        print(f"[ERROR] run_detection: Page {page_num} not found in metadata.")
        raise HTTPException(status_code=404, detail=f"Page {page_num} not found in project.")
        
    target_image_path = os.path.join(project_path, target_page["image_name"])
    
    # 4. Compile Few-Shot History from other corrected pages in the same project.
    few_shot_history = []
    print(f"[DEBUG] run_detection: Gathering few-shot learning history from other user-corrected pages...")
    for page in pages:
        if page.get("page_number") == page_num:
            continue
        if page.get("user_corrected") == True:
            hist_image_path = os.path.join(project_path, page["image_name"])
            print(f"[DEBUG] run_detection: Page {page['page_number']} qualifies as few-shot example. Adding to query history.")
            few_shot_history.append({
                "image_path": hist_image_path,
                "width": page["width"],
                "height": page["height"],
                "windows": page.get("windows", [])
            })
            
    print(f"[DEBUG] run_detection: Compiled {len(few_shot_history)} historical corrected turns.")
    
    # 5. Execute Detection
    try:
        detected_windows = detect_windows(target_image_path, few_shot_history=few_shot_history)
    except Exception as e:
        print(f"[ERROR] run_detection: Gemini service detection call failed. Error: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Gemini API Error: {str(e)}")
        
    # 6. Save detections to metadata.json
    if target_page.get("user_corrected"):
        existing_windows = target_page.get("windows", [])
        merged_windows = _append_non_overlapping_windows(existing_windows, detected_windows)
        target_page["windows"] = merged_windows
        print(f"[DEBUG] run_detection: Preserved {len(existing_windows)} corrected windows and appended {len(merged_windows) - len(existing_windows)} new AI windows.")
    else:
        target_page["windows"] = detected_windows

    try:
        with open(meta_path, "w") as f:
            json.dump(project_meta, f, indent=2)
        print(f"[DEBUG] run_detection: Updated metadata.json successfully with detected window coordinates.")
    except Exception as e:
        print(f"[ERROR] run_detection: Failed to write updated metadata.json. Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save detected window coordinates.")
        
    return {"page_number": page_num, "windows": target_page["windows"]}

@app.get("/api/projects/{project_id}/pages/{page_num}/plan-regions")
async def run_plan_region_detection(project_id: str, page_num: int):
    """
    Detects large floor-plan regions on a sheet so downstream window detection
    can run one plan area at a time.
    """
    print(f"\n[DEBUG] GET /api/projects/{project_id}/pages/{page_num}/plan-regions: Detecting plan regions...")

    project_path = os.path.join(PROJECTS_DIR, project_id)
    meta_path = os.path.join(project_path, "metadata.json")

    if not os.path.exists(meta_path):
        print(f"[ERROR] run_plan_region_detection: Project metadata not found at '{meta_path}'")
        raise HTTPException(status_code=404, detail="Project metadata not found.")

    try:
        with open(meta_path, "r") as f:
            project_meta = json.load(f)
    except Exception as e:
        print(f"[ERROR] run_plan_region_detection: Failed to load metadata.json. Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load project metadata.")

    target_page = next((p for p in project_meta.get("pages", []) if p.get("page_number") == page_num), None)
    if not target_page:
        print(f"[ERROR] run_plan_region_detection: Page {page_num} not found in metadata.")
        raise HTTPException(status_code=404, detail=f"Page {page_num} not found in project.")

    target_image_path = os.path.join(project_path, target_page["image_name"])

    original_pdf_path = os.path.join(project_path, "original.pdf")

    try:
        regions = []
        if os.path.exists(original_pdf_path):
            regions = detect_stantec_plan_regions(
                original_pdf_path,
                page_num,
                page_image_path=target_image_path,
                dpi=PDF_RENDER_DPI,
            )
            if regions:
                print(f"[DEBUG] run_plan_region_detection: Using {len(regions)} Stantec group-home title region(s).")
        if not regions:
            regions = detect_floor_plan_regions(target_image_path)
    except Exception as e:
        print(f"[ERROR] run_plan_region_detection: Failed to detect plan regions. Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to detect plan regions: {str(e)}")

    target_page["plan_regions"] = regions

    try:
        with open(meta_path, "w") as f:
            json.dump(project_meta, f, indent=2)
        print(f"[DEBUG] run_plan_region_detection: Saved {len(regions)} plan region(s).")
    except Exception as e:
        print(f"[ERROR] run_plan_region_detection: Failed to write metadata.json. Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save plan regions.")

    return {"page_number": page_num, "plan_regions": regions}

@app.post("/api/projects/{project_id}/pages/{page_num}/detect-region")
async def run_region_detection(project_id: str, page_num: int, payload: DetectRegionRequest = Body(...)):
    """
    Backward-compatible regional detection endpoint. It delegates to the
    selectable backend endpoint while preserving the older route.
    """
    print(f"\n[DEBUG] POST /api/projects/{project_id}/pages/{page_num}/detect-region: Redirecting to local detector.")
    return await run_local_region_detection(project_id, page_num, payload)

@app.post("/api/projects/{project_id}/pages/{page_num}/detect-local-region")
async def run_local_region_detection(project_id: str, page_num: int, payload: DetectRegionRequest = Body(...)):
    """
    Runs the selected detector backend on a selected plan region and persists
    the result into project metadata.
    """
    backend = payload.backend.strip().lower()
    allowed_backends = {"legacy-local", "deterministic-stantec", "gemini"}
    if backend not in allowed_backends:
        raise HTTPException(status_code=400, detail=f"Unsupported detector backend: {payload.backend}")
    requested_backend = backend

    print(f"\n[DEBUG] POST /api/projects/{project_id}/pages/{page_num}/detect-local-region: Backend={backend} Region={payload.region}")

    project_path = os.path.join(PROJECTS_DIR, project_id)
    meta_path = os.path.join(project_path, "metadata.json")

    if not os.path.exists(meta_path):
        print(f"[ERROR] run_local_region_detection: Project metadata not found at '{meta_path}'")
        raise HTTPException(status_code=404, detail="Project metadata not found.")

    try:
        with open(meta_path, "r") as f:
            project_meta = json.load(f)
    except Exception as e:
        print(f"[ERROR] run_local_region_detection: Failed to load metadata.json. Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load project metadata.")

    target_page = next((p for p in project_meta.get("pages", []) if p.get("page_number") == page_num), None)
    if not target_page:
        print(f"[ERROR] run_local_region_detection: Page {page_num} not found in metadata.")
        raise HTTPException(status_code=404, detail=f"Page {page_num} not found in project.")

    region = payload.region
    if len(region) != 4:
        raise HTTPException(status_code=400, detail="region must be [ymin, xmin, ymax, xmax].")

    ymin, xmin, ymax, xmax = region
    if xmin >= xmax or ymin >= ymax:
        raise HTTPException(status_code=400, detail="Invalid region: zero or negative area.")

    target_image_path = os.path.join(project_path, target_page["image_name"])
    original_pdf_path = os.path.join(project_path, "original.pdf")

    if backend == "deterministic-stantec":
        if not os.path.exists(original_pdf_path):
            print("[WARNING] run_local_region_detection: Original PDF missing, falling back to legacy-local backend.")
            backend = "legacy-local"
        elif not is_stantec_pdf(original_pdf_path, pages=[page_num]):
            print("[DEBUG] run_local_region_detection: PDF does not match Stantec profile, falling back to legacy-local backend.")
            backend = "legacy-local"

    templates = None
    if target_page.get("user_corrected") and target_page.get("windows"):
        corrected_windows = []
        for win in target_page.get("windows", []):
            label = win.get("label")
            if not isinstance(label, str):
                continue
            normalized = label.strip().upper().replace(" ", "")
            if normalized == "W-09" or normalized.endswith("-09") or normalized.endswith("09"):
                corrected_windows.append(win)

        if corrected_windows:
            templates = corrected_windows
            print(f"[DEBUG] run_local_region_detection: Using corrected template windows: {[w['label'] for w in templates]}")
        elif len(target_page.get("windows", [])) == 1:
            templates = target_page.get("windows")
            print("[DEBUG] run_local_region_detection: No W-09 label found, using the single corrected window as template.")

    try:
        if backend == "deterministic-stantec":
            new_windows = detect_stantec_windows_in_region(
                original_pdf_path,
                page_num,
                target_image_path,
                region,
                dpi=PDF_RENDER_DPI,
            )
        elif backend == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key or api_key.strip() == "":
                raise ValueError("Google API Key missing. Please configure GEMINI_API_KEY in your .env file.")

            few_shot_history = []
            for page in project_meta.get("pages", []):
                if page.get("page_number") == page_num:
                    continue
                if page.get("user_corrected") == True:
                    few_shot_history.append({
                        "image_path": os.path.join(project_path, page["image_name"]),
                        "width": page["width"],
                        "height": page["height"],
                        "windows": page.get("windows", []),
                    })
            new_windows = detect_windows_in_region(
                target_image_path,
                region,
                few_shot_history=few_shot_history,
            )
        else:
            new_windows = detect_windows_locally(target_image_path, region=region, templates=templates)
    except Exception as e:
        print(f"[ERROR] run_local_region_detection: {str(e)}")
        status_code = 502 if backend == "gemini" else 500
        raise HTTPException(status_code=status_code, detail=f"{backend} detection error: {str(e)}")

    existing_windows = target_page.get("windows", [])
    if target_page.get("user_corrected"):
        target_page["windows"] = _append_non_overlapping_windows(existing_windows, new_windows)
    else:
        target_page["windows"] = new_windows if payload.replace else existing_windows + new_windows
    target_page["last_detector_backend"] = backend

    try:
        with open(meta_path, "w") as f:
            json.dump(project_meta, f, indent=2)
        action = "Replaced with" if payload.replace else "Appended"
        print(f"[DEBUG] run_local_region_detection: {action} {len(new_windows)} {backend} windows. Total now: {len(target_page['windows'])}")
    except Exception as e:
        print(f"[ERROR] run_local_region_detection: Failed to write metadata.json. Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save local detection windows to project metadata.")

    return {
        "page_number": page_num,
        "requested_backend": requested_backend,
        "backend": backend,
        "new_windows": new_windows,
        "windows": target_page["windows"],
        "total_windows": len(target_page["windows"]),
    }

@app.post("/api/projects/{project_id}/pages/{page_num}/save")
async def save_windows(project_id: str, page_num: int, payload: SaveWindowsRequest = Body(...)):
    """
    Accepts corrected window coordinate data from the web canvas, updates
    metadata.json, and flags the page as 'user_corrected=True'.
    """
    print(f"\n[DEBUG] POST /api/projects/{project_id}/pages/{page_num}/save: Saving window corrections...")
    project_path = os.path.join(PROJECTS_DIR, project_id)
    meta_path = os.path.join(project_path, "metadata.json")
    
    if not os.path.exists(meta_path):
        print(f"[ERROR] save_windows: Project metadata not found at '{meta_path}'")
        raise HTTPException(status_code=404, detail="Project metadata not found.")
        
    try:
        with open(meta_path, "r") as f:
            project_meta = json.load(f)
    except Exception as e:
        print(f"[ERROR] save_windows: Failed to load metadata.json. Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load project metadata.")
        
    target_page = None
    for page in project_meta.get("pages", []):
        if page.get("page_number") == page_num:
            target_page = page
            break
            
    if not target_page:
        print(f"[ERROR] save_windows: Page {page_num} not found in metadata.")
        raise HTTPException(status_code=404, detail=f"Page {page_num} not found in project.")
        
    # Format incoming payload windows
    saved_windows = []
    for win in payload.windows:
        saved_windows.append({
            "label": win.label,
            "box_px": win.box_px
        })

    previous_windows = target_page.get("windows", [])

    target_page["windows"] = saved_windows
    target_page["user_corrected"] = True
    print(f"[DEBUG] save_windows: Updated page {page_num} to user_corrected=True with {len(saved_windows)} window entries.")
    
    try:
        with open(meta_path, "w") as f:
            json.dump(project_meta, f, indent=2)
        print(f"[DEBUG] save_windows: Saved updated metadata.json back to disk.")
        append_correction_log(project_path, {
            "event": "page_windows_saved",
            "project_id": project_id,
            "pdf_name": project_meta.get("pdf_name"),
            "page_number": page_num,
            "image_name": target_page.get("image_name"),
            "width": target_page.get("width"),
            "height": target_page.get("height"),
            "previous_windows": previous_windows,
            "corrected_windows": saved_windows,
            "previous_count": len(previous_windows),
            "corrected_count": len(saved_windows),
        })
        print("[DEBUG] save_windows: Appended correction event to correction_log.jsonl.")
    except Exception as e:
        print(f"[ERROR] save_windows: Failed to save updated metadata.json. Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save window corrections to project metadata.")
        
    return {"status": "success", "saved_count": len(saved_windows)}

@app.post("/api/projects/{project_id}/pages/{page_num}/clear")
async def clear_page(project_id: str, page_num: int):
    """
    Clears saved window annotations and derived plan regions for a page,
    allowing users to reset the page without deleting or reuploading the project.
    """
    print(f"\n[DEBUG] POST /api/projects/{project_id}/pages/{page_num}/clear: Clearing page data")
    project_path = os.path.join(PROJECTS_DIR, project_id)
    meta_path = os.path.join(project_path, "metadata.json")

    if not os.path.exists(meta_path):
        print(f"[ERROR] clear_page: Project metadata not found at '{meta_path}'")
        raise HTTPException(status_code=404, detail="Project metadata not found.")

    try:
        with open(meta_path, "r") as f:
            project_meta = json.load(f)
    except Exception as e:
        print(f"[ERROR] clear_page: Failed to load metadata.json. Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load project metadata.")

    target_page = next((p for p in project_meta.get("pages", []) if p.get("page_number") == page_num), None)
    if not target_page:
        print(f"[ERROR] clear_page: Page {page_num} not found in metadata.")
        raise HTTPException(status_code=404, detail=f"Page {page_num} not found in project.")

    previous_windows = target_page.get("windows", [])
    previous_plan_regions = target_page.get("plan_regions", [])

    target_page["windows"] = []
    target_page["plan_regions"] = []
    target_page["user_corrected"] = False

    try:
        with open(meta_path, "w") as f:
            json.dump(project_meta, f, indent=2)
        append_correction_log(project_path, {
            "event": "page_cleared",
            "project_id": project_id,
            "pdf_name": project_meta.get("pdf_name"),
            "page_number": page_num,
            "image_name": target_page.get("image_name"),
            "cleared_windows": previous_windows,
            "cleared_plan_regions": previous_plan_regions,
            "previous_window_count": len(previous_windows),
            "previous_plan_region_count": len(previous_plan_regions)
        })
        print(f"[DEBUG] clear_page: Cleared page {page_num}, removed {len(previous_windows)} windows and {len(previous_plan_regions)} plan regions.")
    except Exception as e:
        print(f"[ERROR] clear_page: Failed to write metadata.json. Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear page data in project metadata.")

    return {
        "status": "success",
        "page_number": page_num,
        "cleared_windows": len(previous_windows),
        "cleared_plan_regions": len(previous_plan_regions)
    }

# -------------------------------------------------------------
# Static Asset Routing
# -------------------------------------------------------------

# Direct root to index.html
@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/static/index.html")

# Create static directories if they don't exist
os.makedirs("app/static", exist_ok=True)

# Mount the static folder containing HTML/JS/CSS assets
app.mount("/static", StaticFiles(directory="app/static"), name="static")
print("[DEBUG] FastAPI initialization: Static files mounted from 'app/static' to '/static'")
