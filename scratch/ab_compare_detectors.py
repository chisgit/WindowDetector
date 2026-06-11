import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.services.stantec_detector_core import (
    detect_stantec_plan_regions,
    detect_stantec_windows_for_page,
    parse_pages,
    render_page_dpi,
)


def to_xyxy(window: dict) -> list[int]:
    ymin, xmin, ymax, xmax = window["box_px"]
    return [xmin, ymin, xmax, ymax]


def iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def score(pred_boxes: list[list[int]], ref_boxes: list[list[int]], threshold: float = 0.5) -> dict:
    unused = set(range(len(ref_boxes)))
    matches = []
    for pred_idx, pred in enumerate(pred_boxes):
        best = None
        for ref_idx in unused:
            value = iou(pred, ref_boxes[ref_idx])
            if best is None or value > best[0]:
                best = (value, ref_idx)
        if best and best[0] >= threshold:
            unused.remove(best[1])
            matches.append({"pred_idx": pred_idx, "ref_idx": best[1], "iou": best[0]})

    tp = len(matches)
    fp = len(pred_boxes) - tp
    fn = len(ref_boxes) - tp
    precision = tp / len(pred_boxes) if pred_boxes else 0.0
    recall = tp / len(ref_boxes) if ref_boxes else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
    }


def run_standalone(script: Path, pdf: Path, pages_spec: str, outdir: Path, dpi: int, group_homes: list[str] | None) -> None:
    cmd = [
        sys.executable,
        str(script),
        str(pdf),
        "--pages",
        pages_spec,
        "--outdir",
        str(outdir),
        "--dpi",
        str(dpi),
        "--save-raw",
    ]
    if group_homes:
        cmd.extend(["--group-homes", *group_homes])
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def read_standalone_boxes(csv_path: Path, region: dict) -> list[dict]:
    if not csv_path.exists():
        return []

    ymin, xmin, _ymax, _xmax = region["box_px"]
    windows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            x = int(float(row["x_px"]))
            y = int(float(row["y_px"]))
            w = int(float(row["width_px"]))
            h = int(float(row["height_px"]))
            windows.append({
                "label": f"GH{row.get('group_home', region.get('group_home', ''))}-S-{row['id']}",
                "group_home": row.get("group_home", region.get("group_home")),
                "box_px": [ymin + y, xmin + x, ymin + y + h, xmin + x + w],
                "orientation": row.get("orientation"),
                "source": row.get("source"),
                "score": row.get("score"),
            })
    return windows


def draw_overlay(pdf: Path, page_num: int, dpi: int, standalone: list[dict], merged: list[dict], out_path: Path) -> None:
    import cv2

    img, _zoom, _rect = render_page_dpi(pdf, page_num, dpi)
    for window in standalone:
        xmin, ymin, xmax, ymax = to_xyxy(window)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 220, 0), 3)
    for window in merged:
        xmin, ymin, xmax, ymax = to_xyxy(window)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    cv2.imwrite(str(out_path), img)


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B compare standalone v28 and merged Stantec detector outputs.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--pages", default="17")
    parser.add_argument("--group-homes", nargs="*", default=None)
    parser.add_argument("--outdir", type=Path, default=ROOT / "scratch" / "ab_window_detector_output")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--skip-standalone-run", action="store_true")
    parser.add_argument(
        "--standalone-script",
        type=Path,
        default=ROOT / "detectionscript" / "window_detector_unified_profiles_v28_parallel_pane_single_rescue.py",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    standalone_dir = args.outdir / "standalone"
    merged_dir = args.outdir / "merged"
    standalone_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    import fitz

    doc = fitz.open(str(args.pdf))
    pages = parse_pages(args.pages, len(doc))

    if not args.skip_standalone_run:
        run_standalone(args.standalone_script, args.pdf, args.pages, standalone_dir, args.dpi, args.group_homes)

    summary_rows = []
    for page_num in pages:
        regions = detect_stantec_plan_regions(args.pdf, page_num, dpi=args.dpi)
        if args.group_homes:
            regions = [r for r in regions if r.get("group_home") in set(args.group_homes)]

        merged_windows = detect_stantec_windows_for_page(
            args.pdf,
            page_num,
            dpi=args.dpi,
            group_homes=args.group_homes,
        )
        standalone_windows = []
        page_scores = []

        for region in regions:
            gh = region["group_home"]
            standalone_csv = standalone_dir / f"page{page_num}_group_home_{gh}_window_candidates.csv"
            gh_standalone = read_standalone_boxes(standalone_csv, region)
            gh_merged = [w for w in merged_windows if w.get("group_home") == gh]
            standalone_windows.extend(gh_standalone)

            result = score(
                [to_xyxy(w) for w in gh_merged],
                [to_xyxy(w) for w in gh_standalone],
                threshold=args.iou_threshold,
            )
            result.update({
                "page": page_num,
                "group_home": gh,
                "standalone_count": len(gh_standalone),
                "merged_count": len(gh_merged),
            })
            page_scores.append(result)
            summary_rows.append({
                "page": page_num,
                "group_home": gh,
                "standalone_count": len(gh_standalone),
                "merged_count": len(gh_merged),
                "tp": result["tp"],
                "fp": result["fp"],
                "fn": result["fn"],
                "precision": round(result["precision"], 5),
                "recall": round(result["recall"], 5),
                "f1": round(result["f1"], 5),
            })

        overlay_path = args.outdir / f"page{page_num}_comparison_overlay.png"
        draw_overlay(args.pdf, page_num, args.dpi, standalone_windows, merged_windows, overlay_path)

        (merged_dir / f"page{page_num}_merged_windows.json").write_text(json.dumps(merged_windows, indent=2), encoding="utf-8")
        (standalone_dir / f"page{page_num}_standalone_windows_full_page.json").write_text(json.dumps(standalone_windows, indent=2), encoding="utf-8")
        (args.outdir / f"page{page_num}_scorecard.json").write_text(json.dumps(page_scores, indent=2), encoding="utf-8")

    summary_path = args.outdir / "summary_scorecard.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["page", "group_home", "standalone_count", "merged_count", "tp", "fp", "fn", "precision", "recall", "f1"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(summary_path)


if __name__ == "__main__":
    main()
