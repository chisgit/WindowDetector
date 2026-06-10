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
from app.services.pdf_processor import render_pdf_to_images, classify_pages_with_gemini
from app.services.gemini_service import detect_windows, detect_windows_in_region
from app.services.plan_region_detector import detect_floor_plan_regions

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


def append_correction_log(project_path: str, event: dict):
    """
    Stores each user-reviewed correction as JSONL. This creates a trackable
    learning trail without changing the page metadata format.
    """
    log_path = os.path.join(project_path, "correction_log.jsonl")
    event["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

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
        
    # Classify pages as floor plans using Gemini vision (falls back to heuristic if API unavailable)
    api_key = os.getenv("GEMINI_API_KEY", "")
    print(f"[DEBUG] POST /api/upload: Running Gemini floor plan classification on {len(pages_metadata)} pages...")
    classified_pages = classify_pages_with_gemini(pages_metadata, api_key=api_key)

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
    
    # 4. Compile Few-Shot History from previous pages with 'user_corrected=True'
    few_shot_history = []
    print(f"[DEBUG] run_detection: Gathering few-shot learning history from pages prior to page {page_num}...")
    for page in pages:
        # Only check prior pages that have been manually corrected/reviewed
        if page.get("page_number") < page_num and page.get("user_corrected") == True:
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
    target_page["windows"] = detected_windows
    # Keep user_corrected as False until user explicitly clicks Save Corrections
    
    try:
        with open(meta_path, "w") as f:
            json.dump(project_meta, f, indent=2)
        print(f"[DEBUG] run_detection: Updated metadata.json successfully with detected window coordinates.")
    except Exception as e:
        print(f"[ERROR] run_detection: Failed to write updated metadata.json. Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save detected window coordinates.")
        
    return {"page_number": page_num, "windows": detected_windows}

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

    try:
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
    Crops the page image to the provided pixel region and runs Gemini detection
    on just that sub-image. Appends new windows to metadata without replacing
    any existing windows on the page.
    """
    print(f"\n[DEBUG] POST /api/projects/{project_id}/pages/{page_num}/detect-region: Region={payload.region}")

    project_path = os.path.join(PROJECTS_DIR, project_id)
    meta_path = os.path.join(project_path, "metadata.json")

    if not os.path.exists(meta_path):
        print(f"[ERROR] run_region_detection: Project metadata not found at '{meta_path}'")
        raise HTTPException(status_code=404, detail="Project metadata not found.")

    try:
        with open(meta_path, "r") as f:
            project_meta = json.load(f)
    except Exception as e:
        print(f"[ERROR] run_region_detection: Failed to load metadata.json. Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load project metadata.")

    target_page = next((p for p in project_meta.get("pages", []) if p.get("page_number") == page_num), None)
    if not target_page:
        print(f"[ERROR] run_region_detection: Page {page_num} not found in metadata.")
        raise HTTPException(status_code=404, detail=f"Page {page_num} not found in project.")

    region = payload.region
    if len(region) != 4:
        raise HTTPException(status_code=400, detail="region must be [ymin, xmin, ymax, xmax].")

    ymin, xmin, ymax, xmax = region
    if xmin >= xmax or ymin >= ymax:
        raise HTTPException(status_code=400, detail="Invalid region: zero or negative area.")

    target_image_path = os.path.join(project_path, target_page["image_name"])

    # Build few-shot history from prior user-corrected pages.
    few_shot_history = []
    for page in project_meta.get("pages", []):
        if page.get("page_number") < page_num and page.get("user_corrected") == True:
            hist_image_path = os.path.join(project_path, page["image_name"])
            few_shot_history.append({
                "image_path": hist_image_path,
                "width": page["width"],
                "height": page["height"],
                "windows": page.get("windows", [])
            })

    try:
        new_windows = detect_windows_in_region(
            image_path=target_image_path,
            region=region,
            few_shot_history=few_shot_history
        )
    except Exception as e:
        print(f"[ERROR] run_region_detection: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Gemini API Error: {str(e)}")

    existing_windows = target_page.get("windows", [])
    target_page["windows"] = existing_windows + new_windows

    try:
        with open(meta_path, "w") as f:
            json.dump(project_meta, f, indent=2)
        print(f"[DEBUG] run_region_detection: Appended {len(new_windows)} windows. Total now: {len(target_page['windows'])}")
    except Exception as e:
        print(f"[ERROR] run_region_detection: Failed to write metadata.json. Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save detected region windows to project metadata.")

    return {"page_number": page_num, "new_windows": new_windows, "total_windows": len(target_page["windows"])}

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
