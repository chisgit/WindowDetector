import cv2
import numpy as np
from PIL import Image


def _merge_overlapping_boxes(boxes: list[list[int]], overlap_pad: int) -> list[list[int]]:
    merged: list[list[int]] = []

    for box in boxes:
        ymin, xmin, ymax, xmax = box
        did_merge = False
        for existing in merged:
            eymin, exmin, eymax, exmax = existing
            overlaps = not (
                xmax + overlap_pad < exmin
                or xmin - overlap_pad > exmax
                or ymax + overlap_pad < eymin
                or ymin - overlap_pad > eymax
            )
            if overlaps:
                existing[0] = min(eymin, ymin)
                existing[1] = min(exmin, xmin)
                existing[2] = max(eymax, ymax)
                existing[3] = max(exmax, xmax)
                did_merge = True
                break
        if not did_merge:
            merged.append(box[:])

    return merged


def detect_floor_plan_regions(image_path: str, max_regions: int = 12) -> list[dict]:
    """
    Detect large floor-plan drawing regions on a sheet. This is intentionally
    conservative: it finds candidate plan areas so the window detector can run
    one large floor plan at a time instead of on title blocks/schedules/grids.
    """
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    max_dim = 2200
    scale = min(max_dim / max(orig_w, orig_h), 1.0)
    work_w = int(orig_w * scale)
    work_h = int(orig_h * scale)
    work = np.array(img.resize((work_w, work_h), Image.Resampling.LANCZOS))

    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
    # Architectural drawings are mostly dark ink on white paper.
    ink = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)[1]

    # Remove tiny text specks, then dilate enough to connect nearby wall/room
    # linework into one component per plan area.
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    kernel_size = max(10, int(min(work_w, work_h) * 0.012))
    connected = cv2.dilate(ink, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)

    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    page_area = work_w * work_h
    candidates: list[list[int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area_ratio = (w * h) / page_area
        if area_ratio < 0.006 or area_ratio > 0.92:
            continue
        if w < work_w * 0.05 or h < work_h * 0.05:
            continue

        crop_ink = ink[y : y + h, x : x + w]
        ink_density = cv2.countNonZero(crop_ink) / max(1, w * h)
        # Reject near-empty boxes and very dense title/schedule/table regions.
        if ink_density < 0.006 or ink_density > 0.32:
            continue

        pad = max(8, int(min(work_w, work_h) * 0.006))
        candidates.append([
            max(0, y - pad),
            max(0, x - pad),
            min(work_h, y + h + pad),
            min(work_w, x + w + pad),
        ])

    candidates = _merge_overlapping_boxes(candidates, max(6, int(min(work_w, work_h) * 0.006)))

    def score(box: list[int]) -> int:
        ymin, xmin, ymax, xmax = box
        return (ymax - ymin) * (xmax - xmin)

    candidates = sorted(candidates, key=score, reverse=True)[:max_regions]
    candidates = sorted(candidates, key=lambda b: (b[0] // max(1, int(work_h * 0.08)), b[1]))

    regions = []
    for idx, box in enumerate(candidates, start=1):
        ymin, xmin, ymax, xmax = box
        scaled_box = [
            int(round(ymin / scale)),
            int(round(xmin / scale)),
            int(round(ymax / scale)),
            int(round(xmax / scale)),
        ]
        regions.append({
            "label": f"Plan {idx}",
            "box_px": scaled_box,
            "width": scaled_box[3] - scaled_box[1],
            "height": scaled_box[2] - scaled_box[0],
        })

    return regions
