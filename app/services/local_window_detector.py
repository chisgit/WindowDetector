import cv2
import numpy as np
from PIL import Image


def _merge_axis_segments(segments: list[tuple[int, int, int]], orientation: str) -> list[tuple[int, int, int]]:
    merged: list[tuple[int, int, int]] = []
    used = [False] * len(segments)

    for idx, segment in enumerate(segments):
        if used[idx]:
            continue

        group = [segment]
        used[idx] = True
        changed = True

        while changed:
            changed = False
            if orientation == "h":
                start = min(item[0] for item in group)
                axis = int(round(sum(item[1] for item in group) / len(group)))
                end = max(item[2] for item in group)
            else:
                axis = int(round(sum(item[0] for item in group) / len(group)))
                start = min(item[1] for item in group)
                end = max(item[2] for item in group)

            for other_idx, other in enumerate(segments):
                if used[other_idx]:
                    continue

                if orientation == "h":
                    other_start, other_axis, other_end = other
                else:
                    other_axis, other_start, other_end = other

                if abs(other_axis - axis) <= 3 and not (other_start > end + 8 or other_end < start - 8):
                    group.append(other)
                    used[other_idx] = True
                    changed = True

        if orientation == "h":
            merged.append((
                min(item[0] for item in group),
                int(round(sum(item[1] for item in group) / len(group))),
                max(item[2] for item in group),
            ))
        else:
            merged.append((
                int(round(sum(item[0] for item in group) / len(group))),
                min(item[1] for item in group),
                max(item[2] for item in group),
            ))

    return merged


def _cluster_positions(values: list[int], tolerance: int) -> list[tuple[int, int, int]]:
    if not values:
        return []

    values = sorted(values)
    clusters = []
    start = values[0]
    end = values[0]
    members = [values[0]]

    for value in values[1:]:
        if value <= end + tolerance:
            end = value
            members.append(value)
        else:
            clusters.append((start, end, int(round(sum(members) / len(members)))))
            start = value
            end = value
            members = [value]

    clusters.append((start, end, int(round(sum(members) / len(members)))))
    return clusters


def _estimate_building_bbox(horizontal: list[tuple[int, int, int]], vertical: list[tuple[int, int, int]], width: int, height: int) -> list[int]:
    long_vertical_x = [x for x, y1, y2 in vertical if (y2 - y1) >= height * 0.08]
    long_horizontal_y = [y for x1, y, x2 in horizontal if (x2 - x1) >= width * 0.08]

    x_clusters = _cluster_positions(long_vertical_x, max(8, int(width * 0.012)))
    y_clusters = _cluster_positions(long_horizontal_y, max(8, int(height * 0.012)))

    if len(x_clusters) >= 2:
        left = x_clusters[0][0]
        right_idx = len(x_clusters) - 1
        gap_limit = max(45, int(width * 0.035))
        while right_idx > 1 and x_clusters[right_idx][2] - x_clusters[right_idx - 1][2] > gap_limit:
            right_idx -= 1
        right = x_clusters[right_idx][1]
    else:
        left, right = 0, width

    if len(y_clusters) >= 2:
        top = y_clusters[0][0]
        bottom_idx = len(y_clusters) - 1
        gap_limit = max(45, int(height * 0.035))
        while bottom_idx > 1 and y_clusters[bottom_idx][2] - y_clusters[bottom_idx - 1][2] > gap_limit:
            bottom_idx -= 1
        bottom = y_clusters[bottom_idx][1]
    else:
        top, bottom = 0, height

    pad = max(16, int(min(width, height) * 0.012))
    return [
        max(0, top - pad),
        max(0, left - pad),
        min(height, bottom + pad),
        min(width, right + pad),
    ]


def _dedupe_boxes(boxes: list[list[int]], iou_threshold: float = 0.35) -> list[list[int]]:
    def area(box: list[int]) -> int:
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    result: list[list[int]] = []
    for box in sorted(boxes, key=area, reverse=True):
        keep = True
        for existing in result:
            iy1 = max(box[0], existing[0])
            ix1 = max(box[1], existing[1])
            iy2 = min(box[2], existing[2])
            ix2 = min(box[3], existing[3])
            intersection = area([iy1, ix1, iy2, ix2])
            union = area(box) + area(existing) - intersection
            if union and intersection / union > iou_threshold:
                keep = False
                break
        if keep:
            result.append(box)

    return sorted(result, key=lambda b: (b[0], b[1]))


def _detect_horizontal_window_assemblies(
    horizontal: list[tuple[int, int, int]],
    vertical: list[tuple[int, int, int]],
    building_box: list[int],
    perimeter_margin: int,
) -> list[list[int]]:
    """
    Detect the common architectural window symbol shown as several broken,
    nearly-horizontal rail strokes in a narrow band, often with short vertical
    end caps and mullions. This returns one box for the whole assembly.
    """
    useful = []
    for x1, y, x2 in horizontal:
        length = x2 - x1
        if 18 <= length <= 260:
            useful.append((x1, y, x2))

    useful = sorted(useful, key=lambda item: (item[1], item[0]))
    groups: list[list[tuple[int, int, int]]] = []
    for segment in useful:
        placed = False
        for group in groups:
            gy_min = min(item[1] for item in group)
            gy_max = max(item[1] for item in group)
            gx_min = min(item[0] for item in group)
            gx_max = max(item[2] for item in group)
            if abs(segment[1] - ((gy_min + gy_max) / 2)) <= 42 and not (segment[0] > gx_max + 90 or segment[2] < gx_min - 90):
                group.append(segment)
                placed = True
                break
        if not placed:
            groups.append([segment])

    building_ymin, building_xmin, building_ymax, building_xmax = building_box
    boxes: list[list[int]] = []
    for group in groups:
        if len(group) < 4:
            continue

        y_min = min(item[1] for item in group)
        y_max = max(item[1] for item in group)
        x_min = min(item[0] for item in group)
        x_max = max(item[2] for item in group)
        width = x_max - x_min
        height = y_max - y_min
        if not (120 <= width <= 300 and 18 <= height <= 115):
            continue

        cy = (y_min + y_max) / 2
        cx = (x_min + x_max) / 2
        if not (building_xmin <= cx <= building_xmax and building_ymin <= cy <= building_ymax):
            continue
        if min(abs(cx - building_xmin), abs(cx - building_xmax), abs(cy - building_ymin), abs(cy - building_ymax)) > perimeter_margin:
            continue

        caps = [
            item for item in vertical
            if x_min - 18 <= item[0] <= x_max + 18
            and not (item[2] < y_min - 14 or item[1] > y_max + 14)
            and 10 <= (item[2] - item[1]) <= 95
        ]
        left_caps = [item for item in caps if abs(item[0] - x_min) <= 35]
        right_caps = [item for item in caps if abs(item[0] - x_max) <= 35]
        mullions = [item for item in caps if x_min + 35 < item[0] < x_max - 35]

        # Some plans have visible caps; others have a clear middle mullion plus
        # broken rails. Require at least one of those assembly cues.
        if not ((left_caps and right_caps) or mullions or len(caps) >= 2):
            continue

        pad_y = 8
        boxes.append([y_min - pad_y, x_min, y_max + pad_y, x_max])

    return _dedupe_boxes(boxes, iou_threshold=0.2)


def _merge_nearby_window_fragments(boxes: list[list[int]]) -> list[list[int]]:
    merged = [box[:] for box in boxes]
    changed = True

    while changed:
        changed = False
        next_boxes: list[list[int]] = []
        used = [False] * len(merged)

        for idx, box in enumerate(merged):
            if used[idx]:
                continue

            group = [box]
            used[idx] = True
            for other_idx, other in enumerate(merged):
                if used[other_idx]:
                    continue

                y_overlap = min(box[2], other[2]) - max(box[0], other[0])
                x_overlap = min(box[3], other[3]) - max(box[1], other[1])
                y_gap = max(0, max(box[0], other[0]) - min(box[2], other[2]))
                x_gap = max(0, max(box[1], other[1]) - min(box[3], other[3]))
                same_band = y_overlap > 0 or y_gap <= 28
                same_column = x_overlap > 0 or x_gap <= 28

                if same_band and same_column:
                    group.append(other)
                    used[other_idx] = True
                    changed = True

            next_boxes.append([
                min(item[0] for item in group),
                min(item[1] for item in group),
                max(item[2] for item in group),
                max(item[3] for item in group),
            ])

        merged = next_boxes

    filtered = []
    for box in merged:
        height = box[2] - box[0]
        width = box[3] - box[1]
        aspect = max(width, height) / max(1, min(width, height))
        if width < 45 or height < 20:
            continue
        if width > 360 or height > 180:
            continue
        if aspect < 1.4:
            continue
        filtered.append(box)

    return _dedupe_boxes(filtered, iou_threshold=0.25)


def _extract_template_edges(gray: np.ndarray, box: list[int]) -> np.ndarray | None:
    ymin, xmin, ymax, xmax = box
    ymin = max(0, ymin)
    xmin = max(0, xmin)
    ymax = max(0, ymax)
    xmax = max(0, xmax)
    if ymin >= ymax or xmin >= xmax:
        return None

    tpl = gray[ymin:ymax, xmin:xmax]
    if tpl.size == 0 or tpl.shape[0] < 16 or tpl.shape[1] < 16:
        return None

    edges = cv2.Canny(tpl, 50, 150, apertureSize=3)
    if np.count_nonzero(edges) < 25:
        return None
    return edges


def _collect_template_matches(
    result: np.ndarray,
    tpl_w: int,
    tpl_h: int,
    offset_x: int,
    offset_y: int,
    score_threshold: float,
) -> list[list[int]]:
    if result is None or result.size == 0:
        return []

    kernel = np.ones((3, 3), np.uint8)
    local_max = cv2.dilate(result, kernel)
    peaks = np.where((result == local_max) & (result >= score_threshold))
    matches = [
        (float(result[y, x]), x, y)
        for y, x in zip(*peaks)
    ]
    matches.sort(reverse=True)

    selected: list[list[int]] = []
    for score, x, y in matches:
        box = [y + offset_y, x + offset_x, y + tpl_h + offset_y, x + tpl_w + offset_x]
        keep = True
        for existing in selected:
            iy1 = max(box[0], existing[0])
            ix1 = max(box[1], existing[1])
            iy2 = min(box[2], existing[2])
            ix2 = min(box[3], existing[3])
            inter = max(0, iy2 - iy1) * max(0, ix2 - ix1)
            area_box = (box[2] - box[0]) * (box[3] - box[1])
            area_exist = (existing[2] - existing[0]) * (existing[3] - existing[1])
            union = area_box + area_exist - inter
            if union and inter / union > 0.25:
                keep = False
                break
        if keep:
            selected.append(box)
            if len(selected) >= 16:
                break

    return selected


def _find_similar_windows_by_template(
    image: Image.Image,
    templates: list[list[int]],
    region: list[int] | None = None,
    match_threshold: float = 0.62,
) -> list[list[int]]:
    if not templates:
        return []

    full_arr = np.array(image.convert("RGB"))
    gray_full = cv2.cvtColor(full_arr, cv2.COLOR_RGB2GRAY)

    if region:
        ymin, xmin, ymax, xmax = region
        search_arr = gray_full[ymin:ymax, xmin:xmax]
        offset_y = ymin
        offset_x = xmin
    else:
        search_arr = gray_full
        offset_y = 0
        offset_x = 0

    search_edges = cv2.Canny(search_arr, 50, 150, apertureSize=3)
    candidates: list[list[int]] = []

    for box in templates:
        tpl_edges = _extract_template_edges(gray_full, box)
        if tpl_edges is None:
            continue

        for scale in (0.9, 0.96, 1.0, 1.04, 1.1):
            tpl_h = max(12, int(round(tpl_edges.shape[0] * scale)))
            tpl_w = max(12, int(round(tpl_edges.shape[1] * scale)))
            if tpl_h >= search_edges.shape[0] or tpl_w >= search_edges.shape[1]:
                continue

            scaled_tpl = cv2.resize(tpl_edges, (tpl_w, tpl_h), interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(search_edges, scaled_tpl, cv2.TM_CCOEFF_NORMED)
            candidates.extend(_collect_template_matches(
                result,
                tpl_w,
                tpl_h,
                offset_x,
                offset_y,
                score_threshold=match_threshold,
            ))

            if len(candidates) > 120:
                break
        if len(candidates) > 120:
            break

    if not candidates:
        return []

    # Keep the strongest distinct matches only.
    return _dedupe_boxes(candidates, iou_threshold=0.25)


def detect_windows_locally(
    image_path: str,
    region: list[int] | None = None,
    templates: list[dict] | None = None,
) -> list[dict]:
    """
    Local deterministic first pass. It detects paired short horizontal/vertical
    line segments near the exterior building boundary inside the selected plan.
    """
    image = Image.open(image_path).convert("RGB")
    offset_y = 0
    offset_x = 0
    if region:
        ymin, xmin, ymax, xmax = region
        image = image.crop((xmin, ymin, xmax, ymax))
        offset_y = ymin
        offset_x = xmin

    arr = np.array(image)
    height, width = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=25, maxLineGap=6)

    horizontal: list[tuple[int, int, int]] = []
    vertical: list[tuple[int, int, int]] = []
    if lines is not None:
        for raw_line in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(value) for value in raw_line]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dx > dy * 6 and 25 <= dx <= max(320, width * 0.35):
                horizontal.append((min(x1, x2), int(round((y1 + y2) / 2)), max(x1, x2)))
            elif dy > dx * 6 and 25 <= dy <= max(320, height * 0.35):
                vertical.append((int(round((x1 + x2) / 2)), min(y1, y2), max(y1, y2)))

    horizontal = _merge_axis_segments(horizontal, "h")
    vertical = _merge_axis_segments(vertical, "v")
    building_ymin, building_xmin, building_ymax, building_xmax = _estimate_building_bbox(horizontal, vertical, width, height)
    perimeter_margin = max(65, int(min(width, height) * 0.055))
    max_window_length = max(90, int(min(width, height) * 0.09))

    candidate_boxes: list[list[int]] = []

    for i, first in enumerate(vertical):
        ax, ay1, ay2 = first
        alen = ay2 - ay1
        for bx, by1, by2 in vertical[i + 1:]:
            blen = by2 - by1
            gap = abs(bx - ax)
            oy1 = max(ay1, by1)
            oy2 = min(ay2, by2)
            overlap = oy2 - oy1
            if not (8 <= gap <= 45 and 30 <= overlap <= max_window_length and abs(alen - blen) <= 120):
                continue

            cy = (oy1 + oy2) / 2
            cx = (ax + bx) / 2
            if not (building_xmin <= cx <= building_xmax and building_ymin <= cy <= building_ymax):
                continue
            if min(abs(cx - building_xmin), abs(cx - building_xmax), abs(cy - building_ymin), abs(cy - building_ymax)) > perimeter_margin:
                continue

            candidate_boxes.append([oy1, min(ax, bx), oy2, max(ax, bx)])

    for i, first in enumerate(horizontal):
        ax1, ay, ax2 = first
        alen = ax2 - ax1
        for bx1, by, bx2 in horizontal[i + 1:]:
            blen = bx2 - bx1
            gap = abs(by - ay)
            ox1 = max(ax1, bx1)
            ox2 = min(ax2, bx2)
            overlap = ox2 - ox1
            if not (8 <= gap <= 45 and 30 <= overlap <= max_window_length and abs(alen - blen) <= 120):
                continue

            cy = (ay + by) / 2
            cx = (ox1 + ox2) / 2
            if not (building_xmin <= cx <= building_xmax and building_ymin <= cy <= building_ymax):
                continue
            if min(abs(cx - building_xmin), abs(cx - building_xmax), abs(cy - building_ymin), abs(cy - building_ymax)) > perimeter_margin:
                continue

            candidate_boxes.append([min(ay, by), ox1, max(ay, by), ox2])

    assembly_boxes = _detect_horizontal_window_assemblies(
        horizontal,
        vertical,
        [building_ymin, building_xmin, building_ymax, building_xmax],
        perimeter_margin,
    )
    deduped = _dedupe_boxes(
        assembly_boxes + _merge_nearby_window_fragments(_dedupe_boxes(candidate_boxes)),
        iou_threshold=0.25,
    )
    windows = []
    for idx, box in enumerate(deduped, start=1):
        windows.append({
            "label": f"W-{idx:02d}",
            "box_px": [box[0] + offset_y, box[1] + offset_x, box[2] + offset_y, box[3] + offset_x],
        })

    if templates:
        template_boxes = [t["box_px"] for t in templates if isinstance(t.get("box_px"), list) and len(t["box_px"]) == 4]
        if template_boxes:
            similar_boxes = _find_similar_windows_by_template(
                Image.open(image_path).convert("RGB"),
                template_boxes,
                region=region,
            )
            if similar_boxes:
                windows = []
                for idx, box in enumerate(similar_boxes, start=1):
                    windows.append({
                        "label": f"W-{idx:02d}",
                        "box_px": box,
                    })
                return windows

    return windows
