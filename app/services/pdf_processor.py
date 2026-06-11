import os
import io
import fitz  # PyMuPDF
from PIL import Image


def render_pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 200) -> list[dict]:
    """
    Renders each page of a PDF document to two PNG images:
    1. A high-resolution version (rendered at the specified DPI) for window detection and visualization.
    2. A low-resolution thumbnail version (rendered at 50 DPI) for sidebar preview.

    Returns a list of dictionaries containing page metadata.
    """
    print(f"[DEBUG] render_pdf_to_images: Starting rendering for PDF at '{pdf_path}'")
    print(f"[DEBUG] render_pdf_to_images: Target output directory: '{output_dir}', Render DPI: {dpi}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"[DEBUG] render_pdf_to_images: Created output directory '{output_dir}'")

    metadata_list = []

    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"[DEBUG] render_pdf_to_images: Opened PDF successfully. Total pages = {total_pages}")

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            print(f"[DEBUG] render_pdf_to_images: Processing page {page_num}/{total_pages}...")

            page = doc[page_idx]

            # 1. Render High-Resolution Image for Canvas / Detection
            zoom_factor = dpi / 72.0
            matrix = fitz.Matrix(zoom_factor, zoom_factor)

            print(f"[DEBUG] render_pdf_to_images: Rendering high-res image (Page {page_num}) at zoom factor {zoom_factor:.4f} ({dpi} DPI)")
            pix_high = page.get_pixmap(matrix=matrix, alpha=False)

            high_res_name = f"page_{page_num}.png"
            high_res_path = os.path.join(output_dir, high_res_name)
            pix_high.save(high_res_path)
            print(f"[DEBUG] render_pdf_to_images: Saved high-res image to '{high_res_path}' (size: {pix_high.width}x{pix_high.height} px)")

            # 2. Render Low-Resolution Thumbnail for Sidebar (50 DPI)
            thumb_zoom = 50.0 / 72.0
            thumb_matrix = fitz.Matrix(thumb_zoom, thumb_zoom)
            pix_thumb = page.get_pixmap(matrix=thumb_matrix, alpha=False)

            thumb_name = f"page_{page_num}_thumb.png"
            thumb_path = os.path.join(output_dir, thumb_name)
            pix_thumb.save(thumb_path)
            print(f"[DEBUG] render_pdf_to_images: Saved thumbnail image to '{thumb_path}' (size: {pix_thumb.width}x{pix_thumb.height} px)")

            # 3. Collect structural metadata (kept for reference/debugging)
            drawings = page.get_drawings()
            drawings_count = len(drawings)
            text_content = page.get_text()
            text_len = len(text_content.strip())

            print(f"[DEBUG] render_pdf_to_images: Page {page_num} — drawings={drawings_count}, text_len={text_len}")

            metadata_list.append({
                "page_number": page_num,
                "image_name": high_res_name,
                "thumbnail_name": thumb_name,
                "thumbnail_path": thumb_path,   # absolute path for classifier
                "width": pix_high.width,
                "height": pix_high.height,
                "drawings_count": drawings_count,
                "text_length": text_len,
            })

        doc.close()
        print(f"[DEBUG] render_pdf_to_images: Successfully completed processing all {total_pages} pages.")

    except Exception as e:
        print(f"[ERROR] render_pdf_to_images: Failed to process PDF due to error: {str(e)}")
        raise e

    return metadata_list


# -----------------------------------------------------------------
# Gemini-based floor plan classifier
# -----------------------------------------------------------------
def classify_pages_with_gemini(pages_metadata: list[dict], api_key: str) -> list[dict]:
    """
    Uses Gemini 2.5 Flash to classify each page thumbnail as either a floor plan
    or a non-floor-plan (text, cover, schedule, detail, etc.).

    Sends all thumbnails in a single multi-turn prompt to minimise API calls.
    Sets 'is_floor_plan': True/False on each page metadata dict in-place.

    Returns the updated pages_metadata list.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("[ERROR] classify_pages_with_gemini: google-genai package not installed. Falling back to heuristic.")
        return _heuristic_fallback(pages_metadata)

    if not api_key:
        print("[WARNING] classify_pages_with_gemini: No API key provided. Falling back to heuristic.")
        return _heuristic_fallback(pages_metadata)

    print(f"[DEBUG] classify_pages_with_gemini: Classifying {len(pages_metadata)} pages using Gemini 2.5 Flash...")

    client = genai.Client(api_key=api_key)

    # Build a single prompt that lists every page thumbnail with its page number.
    # We ask for a compact JSON array back.
    system_instruction = (
        "You are an expert at reading architectural PDF documents. "
        "You will be shown thumbnail images of pages from a PDF. "
        "For each page, decide if it is an architectural FLOOR PLAN (a drawing showing the layout of rooms, walls, doors, and windows from above). "
        "A floor plan typically contains: room outlines, wall lines, door swings, window symbols, and dimension lines. "
        "Pages that are NOT floor plans include: cover pages, text-only pages, specification sheets, detail drawings, elevation drawings, section drawings, schedules, legends, and title blocks with only text. "
        "Respond ONLY with a JSON array where each element is {\"page\": <page_number>, \"is_floor_plan\": true/false}. "
        "Do not include any explanation."
    )

    # Build contents: interleave thumbnails with page labels
    parts = []
    for meta in pages_metadata:
        thumb_path = meta.get("thumbnail_path", "")
        page_num = meta["page_number"]
        if not os.path.exists(thumb_path):
            print(f"[WARNING] classify_pages_with_gemini: Thumbnail missing for page {page_num} at '{thumb_path}'. Will default to False.")
            continue
        try:
            img = Image.open(thumb_path)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            img_bytes = buf.getvalue()
            parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
            parts.append(types.Part.from_text(text=f"Page {page_num}"))
        except Exception as img_err:
            print(f"[WARNING] classify_pages_with_gemini: Failed to load thumbnail for page {page_num}: {img_err}")

    if not parts:
        print("[WARNING] classify_pages_with_gemini: No thumbnails could be loaded. Falling back to heuristic.")
        return _heuristic_fallback(pages_metadata)

    parts.append(types.Part.from_text(
        text="For each of the pages shown above, is it a floor plan? Reply ONLY with the JSON array."
    ))

    contents = [types.Content(role="user", parts=parts)]

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        temperature=0.0,
    )

    try:
        import time
        start = time.time()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=config,
        )
        latency = time.time() - start
        print(f"[DEBUG] classify_pages_with_gemini: Gemini responded in {latency:.2f}s. Raw: {response.text[:300]}")

        import json
        results = json.loads(response.text)

        # Build lookup map: page_number -> is_floor_plan
        result_map = {item["page"]: item["is_floor_plan"] for item in results}
        print(f"[DEBUG] classify_pages_with_gemini: Classification result map: {result_map}")

        for meta in pages_metadata:
            pn = meta["page_number"]
            if pn in result_map:
                meta["is_floor_plan"] = bool(result_map[pn])
            else:
                # Gemini didn't classify this page — default safe: False
                print(f"[WARNING] classify_pages_with_gemini: Page {pn} missing from Gemini result. Defaulting to False.")
                meta["is_floor_plan"] = False
            print(f"[DEBUG] classify_pages_with_gemini: Page {pn} -> is_floor_plan={meta['is_floor_plan']}")

    except Exception as e:
        print(f"[ERROR] classify_pages_with_gemini: Classification failed: {e}. Falling back to heuristic.")
        return _heuristic_fallback(pages_metadata)

    return pages_metadata


def _heuristic_fallback(pages_metadata: list[dict]) -> list[dict]:
    """
    Simple heuristic fallback used when Gemini classification is unavailable.
    Works reasonably for vector PDFs but is unreliable for scanned PDFs.
    """
    print("[DEBUG] _heuristic_fallback: Running heuristic floor plan classification.")
    for meta in pages_metadata:
        text_len = meta.get("text_length", 0)
        drawings_count = meta.get("drawings_count", 0)
        # Conservative: only mark as floor plan if high drawing count OR low text
        is_floor = not (text_len > 1500 and drawings_count < 10)
        meta["is_floor_plan"] = is_floor
        print(f"[DEBUG] _heuristic_fallback: Page {meta['page_number']} -> is_floor_plan={is_floor} (text={text_len}, drawings={drawings_count})")
    return pages_metadata


def classify_pages_locally(pages_metadata: list[dict]) -> list[dict]:
    """
    Local-only page classification used by the app upload path. It avoids
    external model quota/availability so uploads remain reliable.
    """
    return _heuristic_fallback(pages_metadata)


def is_floor_plan_page(page_metadata: dict) -> bool:
    """
    Backward-compatible single-page floor-plan heuristic used by scratch tests
    and quick local checks.
    """
    text_len = page_metadata.get("text_length", 0)
    drawings_count = page_metadata.get("drawings_count", 0)
    return not (text_len > 1500 and drawings_count < 10)
