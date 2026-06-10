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
    "You are an expert architectural blueprint analyzer. "
    "Your task is to identify actual window openings on the provided floor plan. "
    "A valid window must be embedded in a wall band and bounded by two perpendicular cap ends. "
    "First find horizontal and vertical wall bands, then find openings within those walls. "
    "For a vertical window, the top and bottom boundaries are horizontal cap lines; return a tight box from top cap to bottom cap while staying inside the wall thickness. "
    "For a horizontal window, the left and right boundaries are vertical cap lines; return a tight box from left cap to right cap while staying inside the wall thickness. "
    "The box must land on the wall, not beside it, and must not include room labels, dimensions, colored annotation labels, or extra wall continuation beyond the caps. "
    "Reject candidates that do not have paired cap ends, are not between wall lines, or are doors, cabinets, plumbing/fixture symbols, text, dimension lines, revision clouds, or furniture. "
    "If a page already contains colored training markup, ignore colored label rectangles and use only the underlying architectural window cap geometry. "
    "Count and label accepted windows sequentially (e.g. W-01, W-02) resetting on each floor plan. "
    "If you can clearly read the room name adjacent to the accepted wall opening, include it in the label (e.g. Living Room W-01, Bed 1 W-02). "
    "Return bounding box coordinates in normalized integer scale [ymin, xmin, ymax, xmax] from 0 to 1000. "
    "Respond ONLY with a valid JSON object in this exact format: "
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

    # 7. Call Gemini with exponential backoff retry logic
    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=generation_config,
            )
            latency = time.time() - start_time
            print(f"[DEBUG] detect_windows: API request succeeded in {latency:.2f} seconds.")

            # Parse response
            response_text = response.text
            print(f"[DEBUG] detect_windows: Raw response string (first 500 chars): {response_text[:500]}")

            data = json.loads(response_text)
            detected_windows = data.get("windows", [])
            print(f"[DEBUG] detect_windows: Parsed {len(detected_windows)} windows from Gemini output.")

            # Translate normalized coordinates back to original image pixels
            final_windows = []
            for win in detected_windows:
                box_2d = win.get("box_2d", [])
                if len(box_2d) == 4:
                    ymin, xmin, ymax, xmax = box_2d
                    ymin_px = int((ymin / 1000.0) * orig_height)
                    xmin_px = int((xmin / 1000.0) * orig_width)
                    ymax_px = int((ymax / 1000.0) * orig_height)
                    xmax_px = int((xmax / 1000.0) * orig_width)

                    final_windows.append({
                        "label": win.get("label", "W"),
                        "box_px": [ymin_px, xmin_px, ymax_px, xmax_px]
                    })
                else:
                    print(f"[WARNING] detect_windows: Skipping malformed box_2d entry: {box_2d}")

            print(f"[DEBUG] detect_windows: Bounding boxes translated to original dimensions successfully. Returning {len(final_windows)} windows.")
            return final_windows

        except Exception as e:
            print(f"[WARNING] detect_windows: Attempt {attempt + 1} failed. Error: {str(e)}")
            if attempt < max_retries - 1:
                print(f"[DEBUG] detect_windows: Waiting {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)
                retry_delay *= 2  # exponential backoff
            else:
                print("[ERROR] detect_windows: All Gemini API retries exhausted.")
                raise e


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

    max_retries = 3
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=generation_config,
            )
            latency = time.time() - start_time
            print(f"[DEBUG] detect_windows_in_region: API request succeeded in {latency:.2f} seconds.")

            response_text = response.text
            print(f"[DEBUG] detect_windows_in_region: Raw response string (first 500 chars): {response_text[:500]}")
            data = json.loads(response_text)
            detected_windows = data.get("windows", [])

            final_windows = []
            for win in detected_windows:
                box_2d = win.get("box_2d", [])
                if len(box_2d) == 4:
                    ymin_norm, xmin_norm, ymax_norm, xmax_norm = box_2d
                    ymin_px = int((ymin_norm / 1000.0) * crop_height) + ymin
                    xmin_px = int((xmin_norm / 1000.0) * crop_width) + xmin
                    ymax_px = int((ymax_norm / 1000.0) * crop_height) + ymin
                    xmax_px = int((xmax_norm / 1000.0) * crop_width) + xmin
                    final_windows.append({
                        "label": win.get("label", "W"),
                        "box_px": [ymin_px, xmin_px, ymax_px, xmax_px]
                    })
                else:
                    print(f"[WARNING] detect_windows_in_region: Skipping malformed box_2d entry: {box_2d}")

            print(f"[DEBUG] detect_windows_in_region: Returning {len(final_windows)} windows from region crop.")
            return final_windows
        except Exception as e:
            print(f"[WARNING] detect_windows_in_region: Attempt {attempt + 1} failed. Error: {str(e)}")
            if attempt < max_retries - 1:
                print(f"[DEBUG] detect_windows_in_region: Waiting {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("[ERROR] detect_windows_in_region: All Gemini API retries exhausted.")
                raise e


def clip_val(val, min_val, max_val):
    return max(min(val, max_val), min_val)
