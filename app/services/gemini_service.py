import os
import time
import json
import io
from PIL import Image
from google import genai
from google.genai import types
from pydantic import BaseModel

# -------------------------------------------------------------
# Gemini Vision Detection Service
# Using google.genai SDK with gemini-2.5-flash
# -------------------------------------------------------------

SYSTEM_INSTRUCTION = (
    "You are a vision expert for a window-covering business that bids from architectural floor plans. "
    "Your job is to find actual window openings so their boxes can later be scaled into real-world widths. "
    "Use general floor-plan geometry, not memorized coordinates from any one drawing. "
    "First decide whether the image contains a floor plan: look for rectangular room layouts, wall lines, room labels, doors, fixtures, dimensions, and a building perimeter polygon. "
    "Then scan the exterior perimeter walls and interior walls for consistently sized wall breaks or architectural window symbols. "
    "Do not annotate repeated drawing panels, schedules, title blocks, room-name rectangles, legends, tables, specification grids, or demo-plan grids. "
    "If the page contains multiple separate floor plans, only return windows that you can tie to clear wall/cap geometry; do not fill every repeated panel. "
    "A valid window must be embedded in a wall band, between wall lines, and bounded by two perpendicular cap ends. "
    "On a north/south vertical wall, detect the horizontal cap lines that close the top and bottom of the window; draw the box from top cap to bottom cap and keep it inside the wall thickness. "
    "On an east/west horizontal wall, detect the vertical cap lines that close the left and right of the window; draw the box from left cap to right cap and keep it inside the wall thickness. "
    "For double-line windows, protruding bay-like windows, centerline windows, or two-pane symbols, return one covering box for the whole window assembly from outer cap to outer cap. "
    "Most valid boxes are narrow wall elements: clearly wider-than-tall or taller-than-wide. Be suspicious of square or room-sized rectangles. "
    "Give extra attention to common window locations: bedroom exterior walls, kitchens near sinks, living/dining rooms, offices, tub rooms, and repeated same-size symbols. "
    "Look for repeated window shapes with similar dimensions, especially along the same elevation, but only accept them when they sit in a wall and have cap evidence. "
    "If the plan is small or dense, reason as though zooming into the wall bands and cap ends; precision at the cap ends matters more than label placement. "
    "Reject doors and openings with swing arcs, room entryways, cabinets, plumbing/fixture symbols, furniture, text, dimension lines, revision clouds, section/elevation markers, and colored training labels. "
    "The returned box must land on the wall itself, not beside it, and must not include room labels, dimensions, colored annotation labels, or wall continuation beyond the caps. "
    "If a page already contains colored training markup, ignore colored label rectangles and use only the underlying architectural window geometry. "
    "Count and label accepted windows sequentially (e.g. W-01, W-02) resetting on each floor plan. "
    "If you can clearly read the adjacent room name, include it in the label (e.g. Living Room W-01, Bedroom W-02, Kitchen W-03). "
    "Return bounding box coordinates in normalized integer scale [ymin, xmin, ymax, xmax] from 0 to 1000. "
    "Respond only with a valid JSON object in this exact format: "
    "{\"windows\": [{\"label\": \"W-01\", \"box_2d\": [ymin, xmin, ymax, xmax]}, ...]}"
)

JSON_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "windows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4
                    }
                },
                "required": ["label", "box_2d"]
            }
        }
    },
    "required": ["windows"]
}


class WindowDetectionItem(BaseModel):
    label: str
    box_px: list[int]


class WindowDetectionResponse(BaseModel):
    windows: list[WindowDetectionItem]


def _pil_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    """Convert a PIL Image to raw bytes."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _model_candidates() -> list[str]:
    """
    Ordered Gemini model fallback list. Override with GEMINI_MODELS as a
    comma-separated list, or GEMINI_MODEL for a single preferred model.
    """
    configured_models = os.getenv("GEMINI_MODELS", "").strip()
    if configured_models:
        return [model.strip() for model in configured_models.split(",") if model.strip()]

    configured_model = os.getenv("GEMINI_MODEL", "").strip()
    if configured_model:
        return [configured_model]

    return [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
    ]


def _generate_content_with_fallback(client, contents, generation_config, caller_name: str):
    """
    Calls Gemini with model fallback and exponential backoff. This is mainly to
    ride through temporary 503 high-demand errors without forcing a user retry.
    """
    max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "4"))
    initial_retry_delay = float(os.getenv("GEMINI_RETRY_DELAY_SECONDS", "3"))
    last_error = None

    for model in _model_candidates():
        retry_delay = initial_retry_delay
        for attempt in range(max_retries):
            try:
                print(f"[DEBUG] {caller_name}: Calling model '{model}' attempt {attempt + 1}/{max_retries}.")
                start_time = time.time()
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=generation_config,
                )
                latency = time.time() - start_time
                print(f"[DEBUG] {caller_name}: Model '{model}' succeeded in {latency:.2f} seconds.")
                return response
            except Exception as e:
                last_error = e
                err_text = str(e)
                print(f"[WARNING] {caller_name}: Model '{model}' attempt {attempt + 1} failed. Error: {err_text}")

                if attempt < max_retries - 1:
                    print(f"[DEBUG] {caller_name}: Waiting {retry_delay:.1f} seconds before retrying '{model}'...")
                    time.sleep(retry_delay)
                    retry_delay *= 2

        print(f"[WARNING] {caller_name}: Exhausted retries for model '{model}'. Trying next fallback if available.")

    print(f"[ERROR] {caller_name}: All Gemini model fallbacks exhausted.")
    raise last_error


def _convert_and_filter_windows(detected_windows: list, orig_width: int, orig_height: int, caller_name: str) -> list[dict]:
    """
    Convert normalized Gemini boxes to pixel boxes and reject obvious floods:
    square blocks, huge page regions, malformed boxes, and runaway counts.
    """
    raw_count = len(detected_windows)
    max_windows = int(os.getenv("GEMINI_MAX_WINDOWS", "120"))
    min_aspect_ratio = float(os.getenv("WINDOW_MIN_ASPECT_RATIO", "1.45"))
    max_width_ratio = float(os.getenv("WINDOW_MAX_WIDTH_RATIO", "0.18"))
    max_height_ratio = float(os.getenv("WINDOW_MAX_HEIGHT_RATIO", "0.18"))
    max_area_ratio = float(os.getenv("WINDOW_MAX_AREA_RATIO", "0.01"))

    if raw_count > max_windows * 5:
        print(f"[WARNING] {caller_name}: Model returned {raw_count} boxes, likely grid/table hallucination. Rejecting output.")
        return []

    final_windows = []
    rejected = 0
    page_area = orig_width * orig_height

    for win in detected_windows:
        box_2d = win.get("box_2d", [])
        if len(box_2d) != 4:
            print(f"[WARNING] {caller_name}: Skipping malformed box_2d entry: {box_2d}")
            rejected += 1
            continue

        ymin, xmin, ymax, xmax = box_2d
        ymin_px = int(clip_val((ymin / 1000.0) * orig_height, 0, orig_height))
        xmin_px = int(clip_val((xmin / 1000.0) * orig_width, 0, orig_width))
        ymax_px = int(clip_val((ymax / 1000.0) * orig_height, 0, orig_height))
        xmax_px = int(clip_val((xmax / 1000.0) * orig_width, 0, orig_width))

        if ymin_px > ymax_px:
            ymin_px, ymax_px = ymax_px, ymin_px
        if xmin_px > xmax_px:
            xmin_px, xmax_px = xmax_px, xmin_px

        width = xmax_px - xmin_px
        height = ymax_px - ymin_px
        if width < 6 or height < 6:
            rejected += 1
            continue

        aspect_ratio = max(width, height) / max(1, min(width, height))
        area_ratio = (width * height) / page_area

        if aspect_ratio < min_aspect_ratio:
            rejected += 1
            continue
        if width / orig_width > max_width_ratio or height / orig_height > max_height_ratio:
            rejected += 1
            continue
        if area_ratio > max_area_ratio:
            rejected += 1
            continue

        final_windows.append({
            "label": win.get("label", f"W-{len(final_windows) + 1:02d}"),
            "box_px": [ymin_px, xmin_px, ymax_px, xmax_px]
        })

    if len(final_windows) > max_windows:
        print(f"[WARNING] {caller_name}: Filtered count {len(final_windows)} still exceeds {max_windows}. Rejecting output.")
        return []

    print(f"[DEBUG] {caller_name}: Accepted {len(final_windows)} windows; rejected {rejected} of {raw_count} candidates.")
    return final_windows


def detect_windows(image_path: str, few_shot_history: list = None) -> list[dict]:
    """
    Calls the Gemini 2.0 Flash Multimodal API to detect windows in the given floor plan image.
    Supports few-shot visual history from corrected pages of the same project.

    few_shot_history: List of dictionaries matching:
        {
            "image_path": str,
            "width": int,
            "height": int,
            "windows": [
                {"label": "W-01", "box_px": [ymin, xmin, ymax, xmax]}
            ]
        }

    Returns: list of {"label": str, "box_px": [ymin_px, xmin_px, ymax_px, xmax_px]}
    """
    print(f"[DEBUG] detect_windows: Initializing window detection for image '{image_path}'")

    # 1. Fetch and validate API Key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] detect_windows: GEMINI_API_KEY environment variable is missing.")
        raise ValueError("Gemini API key is not configured in environment variables. Please check your .env file.")

    # 2. Initialize the new google.genai client
    client = genai.Client(api_key=api_key)
    print("[DEBUG] detect_windows: google.genai Client initialized successfully.")

    # 3. Open Target Image and get original dimensions
    if not os.path.exists(image_path):
        print(f"[ERROR] detect_windows: Target image not found at '{image_path}'")
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img = Image.open(image_path)
    orig_width, orig_height = img.size
    print(f"[DEBUG] detect_windows: Opened target image. Size: {orig_width}x{orig_height} px")

    # 4. Create a scaled-down image copy if it exceeds 1600px in either dimension to save bandwidth/tokens.
    max_dim = 1600
    scaled_img = img.copy()
    if orig_width > max_dim or orig_height > max_dim:
        scaled_img.thumbnail((max_dim, max_dim))
        print(f"[DEBUG] detect_windows: Scaled target image copy for API request to {scaled_img.size[0]}x{scaled_img.size[1]} px")

    # 5. Construct the contents list (system instruction + few-shot turns + query image)
    contents = []

    # Add few-shot history examples as multi-turn context
    if few_shot_history:
        print(f"[DEBUG] detect_windows: Constructing few-shot visual history with {len(few_shot_history)} page examples.")
        for idx, turn in enumerate(few_shot_history):
            print(f"[DEBUG] detect_windows: Compiling history turn {idx+1} (Image: '{turn['image_path']}')")
            try:
                hist_img = Image.open(turn["image_path"])
                if hist_img.width > max_dim or hist_img.height > max_dim:
                    hist_img.thumbnail((max_dim, max_dim))

                # Normalize pixel coordinates to 0-1000
                hist_windows = []
                hist_w, hist_h = turn["width"], turn["height"]
                for win in turn.get("windows", []):
                    box_px = win["box_px"]
                    ymin_norm = int(clip_val((box_px[0] / hist_h) * 1000, 0, 1000))
                    xmin_norm = int(clip_val((box_px[1] / hist_w) * 1000, 0, 1000))
                    ymax_norm = int(clip_val((box_px[2] / hist_h) * 1000, 0, 1000))
                    xmax_norm = int(clip_val((box_px[3] / hist_w) * 1000, 0, 1000))
                    hist_windows.append({
                        "label": win["label"],
                        "box_2d": [ymin_norm, xmin_norm, ymax_norm, xmax_norm]
                    })

                hist_bytes = _pil_to_bytes(hist_img)
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_bytes(data=hist_bytes, mime_type="image/jpeg"),
                            types.Part.from_text(text="Detect windows on this floor plan and return them in JSON format."),
                        ]
                    )
                )
                contents.append(
                    types.Content(
                        role="model",
                        parts=[
                            types.Part.from_text(text=json.dumps({"windows": hist_windows})),
                        ]
                    )
                )
                print(f"[DEBUG] detect_windows: History turn {idx+1} appended with {len(hist_windows)} corrected window boxes.")
            except Exception as hist_err:
                print(f"[WARNING] detect_windows: Skipping history turn {idx+1} due to loading error: {str(hist_err)}")

    # Append the current page image as the final user turn
    scaled_bytes = _pil_to_bytes(scaled_img)
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=scaled_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text="Detect windows on this floor plan. Return the window bounding boxes in normalized coordinates [ymin, xmin, ymax, xmax] from 0 to 1000."),
            ]
        )
    )

    print(f"[DEBUG] detect_windows: Contents list compiled with {len(contents)} turn(s). Sending request to Gemini API...")

    # 6. Build generation config with JSON response schema
    generation_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=JSON_RESPONSE_SCHEMA,
        temperature=0.1,
    )

    # 7. Call Gemini with model fallback and exponential backoff retry logic
    response = _generate_content_with_fallback(client, contents, generation_config, "detect_windows")

    # Parse response
    response_text = response.text
    print(f"[DEBUG] detect_windows: Raw response string (first 500 chars): {response_text[:500]}")

    data = json.loads(response_text)
    detected_windows = data.get("windows", [])
    print(f"[DEBUG] detect_windows: Parsed {len(detected_windows)} windows from Gemini output.")

    final_windows = _convert_and_filter_windows(detected_windows, orig_width, orig_height, "detect_windows")

    print(f"[DEBUG] detect_windows: Bounding boxes translated to original dimensions successfully. Returning {len(final_windows)} windows.")
    return final_windows


def detect_windows_in_region(image_path: str, region: list, few_shot_history: list = None) -> list[dict]:
    """
    Crop the target page image to the requested region and run Gemini detection
    on that crop. Returned boxes are adjusted to the original full-image coordinate
    space and appended back to the current page's window list.
    """
    print(f"[DEBUG] detect_windows_in_region: Cropping region {region} from '{image_path}'")

    if not os.path.exists(image_path):
        print(f"[ERROR] detect_windows_in_region: Target image not found at '{image_path}'")
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if len(region) != 4:
        raise ValueError("Region must be [ymin, xmin, ymax, xmax].")

    ymin, xmin, ymax, xmax = region
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Invalid region: zero or negative area.")

    img = Image.open(image_path)
    crop = img.crop((xmin, ymin, xmax, ymax))
    crop_width, crop_height = crop.size

    max_dim = 1600
    scaled_crop = crop.copy()
    if crop_width > max_dim or crop_height > max_dim:
        scaled_crop.thumbnail((max_dim, max_dim))
        print(f"[DEBUG] detect_windows_in_region: Scaled crop for API request to {scaled_crop.size[0]}x{scaled_crop.size[1]} px")

    contents = []
    if few_shot_history:
        print(f"[DEBUG] detect_windows_in_region: Constructing few-shot visual history with {len(few_shot_history)} page examples.")
        for idx, turn in enumerate(few_shot_history):
            try:
                hist_img = Image.open(turn["image_path"])
                if hist_img.width > max_dim or hist_img.height > max_dim:
                    hist_img.thumbnail((max_dim, max_dim))

                hist_windows = []
                hist_w, hist_h = turn["width"], turn["height"]
                for win in turn.get("windows", []):
                    box_px = win["box_px"]
                    ymin_norm = int(clip_val((box_px[0] / hist_h) * 1000, 0, 1000))
                    xmin_norm = int(clip_val((box_px[1] / hist_w) * 1000, 0, 1000))
                    ymax_norm = int(clip_val((box_px[2] / hist_h) * 1000, 0, 1000))
                    xmax_norm = int(clip_val((box_px[3] / hist_w) * 1000, 0, 1000))
                    hist_windows.append({
                        "label": win["label"],
                        "box_2d": [ymin_norm, xmin_norm, ymax_norm, xmax_norm]
                    })

                hist_bytes = _pil_to_bytes(hist_img)
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_bytes(data=hist_bytes, mime_type="image/jpeg"),
                            types.Part.from_text(text="Detect windows on this floor plan and return them in JSON format."),
                        ]
                    )
                )
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=json.dumps({"windows": hist_windows}))]
                    )
                )
                print(f"[DEBUG] detect_windows_in_region: History turn {idx+1} appended with {len(hist_windows)} corrected window boxes.")
            except Exception as hist_err:
                print(f"[WARNING] detect_windows_in_region: Skipping history turn {idx+1} due to loading error: {str(hist_err)}")

    scaled_bytes = _pil_to_bytes(scaled_crop)
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=scaled_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text="Detect windows on this cropped floor plan region. Return the window bounding boxes in normalized coordinates [ymin, xmin, ymax, xmax] from 0 to 1000."),
            ]
        )
    )

    print(f"[DEBUG] detect_windows_in_region: Sending region crop to Gemini with {len(contents)} turn(s).")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key is not configured in environment variables. Please check your .env file.")

    client = genai.Client(api_key=api_key)
    generation_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=JSON_RESPONSE_SCHEMA,
        temperature=0.1,
    )

    response = _generate_content_with_fallback(client, contents, generation_config, "detect_windows_in_region")

    response_text = response.text
    print(f"[DEBUG] detect_windows_in_region: Raw response string (first 500 chars): {response_text[:500]}")
    data = json.loads(response_text)
    detected_windows = data.get("windows", [])

    crop_windows = _convert_and_filter_windows(detected_windows, crop_width, crop_height, "detect_windows_in_region")
    final_windows = []
    for win in crop_windows:
        win_ymin, win_xmin, win_ymax, win_xmax = win["box_px"]
        final_windows.append({
            "label": win.get("label", f"W-{len(final_windows) + 1:02d}"),
            "box_px": [win_ymin + ymin, win_xmin + xmin, win_ymax + ymin, win_xmax + xmin]
        })

    print(f"[DEBUG] detect_windows_in_region: Returning {len(final_windows)} windows from region crop.")
    return final_windows


def clip_val(val, min_val, max_val):
    return max(min(val, max_val), min_val)
