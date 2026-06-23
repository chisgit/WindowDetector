"""
Window-detection accuracy loop harness.

Drives app/services/stantec_detector_core.py (the SAME entry points the UI uses),
scores detections against the human-corrected ground truth in the project's
correction_log.jsonl, and reports per-plan F1 with a length+placement metric.

Spec: WINDOW_98_LOOP.json  (NO HARDCODING of fixes — this file only measures.)
Run:  PYTHONPATH=. python3 loop/harness.py [--tol 2] [--overlays]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.services import stantec_detector_core as det  # noqa: E402

PROJECT = ROOT / "projects" / "7dde299f-ec42-4424-8f46-f865bfdb4a3b"
PDF = PROJECT / "original.pdf"
PAGE_IMG = PROJECT / "page_17.png"
PAGE = 17
CORR_LOG = PROJECT / "correction_log.jsonl"
TRAIN_GHS = ["1", "2", "3", "4", "26"]
OUTDIR = ROOT / "loop" / "out"

# Labels the user types onto a box in the correction UI to mark it NOT a window
# (kept as boxes so we keep them as hard negatives, not ground-truth windows).
REJECT_LABEL_KEYS = ("not window", "no caps", "no break", "looks like window", "not a window",
                     "door", "stair")


def _is_reject_label(label):
    s = (label or "").lower()
    return any(k in s for k in REJECT_LABEL_KEYS)


# ---------- geometry helpers (box = [ymin, xmin, ymax, xmax]) ----------
def dims(b):
    h = b[2] - b[0]
    w = b[3] - b[1]
    return w, h


def orientation(b):
    w, h = dims(b)
    return "horizontal" if w >= h else "vertical"


def long_side(b):
    w, h = dims(b)
    return max(w, h)


def center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def corner_offset(p, g):
    """Max absolute per-edge offset between two yxyx boxes."""
    return max(abs(p[0] - g[0]), abs(p[1] - g[1]), abs(p[2] - g[2]), abs(p[3] - g[3]))


def region_contains_center(region_box, b):
    cy, cx = center(b)
    y0, x0, y1, x1 = region_box
    return y0 <= cy <= y1 and x0 <= cx <= x1


# ---------- ground truth ----------
def load_ground_truth():
    """Last page_windows_saved event -> canonical full-page corrected windows."""
    last = None
    for line in CORR_LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        ev = json.loads(line)
        if ev.get("event") == "page_windows_saved" and ev.get("page_number") == PAGE:
            last = ev
    if last is None:
        raise RuntimeError("no page_windows_saved event for page 17")
    seen, boxes = set(), []
    for w in last["corrected_windows"]:
        if _is_reject_label(w.get("label")):
            continue  # user-marked NOT-a-window -> excluded from GT (see load_hard_negatives)
        b = tuple(int(v) for v in w["box_px"])
        if b in seen:
            continue
        seen.add(b)
        boxes.append(list(b))
    return boxes


def load_hard_negatives():
    """Boxes the user relabelled in the UI to explain they are NOT windows
    (e.g. 'no caps', 'no break in the wall'). Gold negatives for the classifier."""
    last = None
    for line in CORR_LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        ev = json.loads(line)
        if ev.get("event") == "page_windows_saved" and ev.get("page_number") == PAGE:
            last = ev
    if last is None:
        return []
    return [list(int(v) for v in w["box_px"])
            for w in last["corrected_windows"] if _is_reject_label(w.get("label"))]


def assign_to_gh(boxes, regions):
    """Assign each box to the group-home region whose center is nearest among
    regions that contain the box center (fallback: globally nearest center)."""
    out = {}
    for b in boxes:
        cy, cx = center(b)
        containing = [r for r in regions if region_contains_center(r["box_px"], b)]
        pool = containing or regions
        best = min(pool, key=lambda r: (
            (center(r["box_px"])[0] - cy) ** 2 + (center(r["box_px"])[1] - cx) ** 2))
        out.setdefault(best["group_home"], []).append(b)
    return out


# ---------- matching ----------
def match(preds, gts, tol):
    """Greedy one-to-one. TP if same orientation and all 4 edges within tol."""
    pairs = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            if orientation(p) != orientation(g):
                continue
            off = corner_offset(p, g)
            if off <= tol:
                pairs.append((off, pi, gi))
    pairs.sort()
    used_p, used_g, tp_pairs = set(), set(), []
    for off, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi); tp_pairs.append((pi, gi, off))
    fp = [i for i in range(len(preds)) if i not in used_p]
    fn = [i for i in range(len(gts)) if i not in used_g]
    return tp_pairs, fp, fn


def f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 1.0


# ---------- main ----------
def dedup(boxes, tol=6):
    """Merge near-identical predictions (same window detected in two overlapping
    crops). Keeps the first; drops any later box whose 4 edges are all within tol."""
    kept = []
    for b in boxes:
        if any(orientation(b) == orientation(k) and corner_offset(b, k) <= tol for k in kept):
            continue
        kept.append(b)
    return kept


def gh_of(b, regions):
    cy, cx = center(b)
    containing = [r for r in regions if region_contains_center(r["box_px"], b)]
    pool = containing or regions
    best = min(pool, key=lambda r: (
        (center(r["box_px"])[0] - cy) ** 2 + (center(r["box_px"])[1] - cx) ** 2))
    return best["group_home"]


def run(tol=2, overlays=False):
    regions = det.detect_stantec_plan_regions(PDF, PAGE, page_image_path=PAGE_IMG)
    gt_all = load_ground_truth()

    # GLOBAL page-level scoring: union of predictions from all trained crops,
    # deduped (overlap bands detect the same window twice), matched vs ALL gt.
    preds_raw = []
    for gh in TRAIN_GHS:
        wins = det.detect_stantec_windows_for_page(
            PDF, PAGE, page_image_path=PAGE_IMG, group_homes=[gh])
        preds_raw.extend(w["box_px"] for w in wins)
    preds = dedup(preds_raw)
    gts = gt_all

    tp_pairs, fp, fn = match(preds, gts, tol)
    tot_tp, tot_fp, tot_fn = len(tp_pairs), len(fp), len(fn)

    # nearest-offset histogram (how close is each pred to ANY same-orient gt)
    offset_hist = {2: 0, 5: 0, 10: 0, 20: 0, 50: 0, "miss": 0}
    for p in preds:
        cands = [corner_offset(p, g) for g in gts if orientation(g) == orientation(p)]
        o = min(cands) if cands else 9999
        for thr in (2, 5, 10, 20, 50):
            if o <= thr:
                offset_hist[thr] += 1
                break
        else:
            offset_hist["miss"] += 1

    # per-plan breakdown (report only) by binning each box to nearest region
    per_gh = {gh: {"pred": 0, "gt": 0, "tp": 0, "fp": 0, "fn": 0,
                   "fp_boxes": [], "fn_boxes": []} for gh in TRAIN_GHS}
    for p in preds:
        g = gh_of(p, regions)
        if g in per_gh:
            per_gh[g]["pred"] += 1
    for b in gts:
        g = gh_of(b, regions)
        if g in per_gh:
            per_gh[g]["gt"] += 1
    for pi, gi, off in tp_pairs:
        g = gh_of(gts[gi], regions)
        if g in per_gh:
            per_gh[g]["tp"] += 1
    for i in fp:
        g = gh_of(preds[i], regions)
        if g in per_gh:
            per_gh[g]["fp"] += 1; per_gh[g]["fp_boxes"].append(preds[i])
    for i in fn:
        g = gh_of(gts[i], regions)
        if g in per_gh:
            per_gh[g]["fn"] += 1; per_gh[g]["fn_boxes"].append(gts[i])
    for g in per_gh:
        d = per_gh[g]
        d["f1"] = round(f1(d["tp"], d["fp"], d["fn"]), 4)

    macro = sum(per_gh[g]["f1"] for g in TRAIN_GHS) / len(TRAIN_GHS)
    report = {
        "tol_px": tol,
        "macro_f1": round(macro, 4),
        "micro_f1": round(f1(tot_tp, tot_fp, tot_fn), 4),
        "totals": {"tp": tot_tp, "fp": tot_fp, "fn": tot_fn,
                   "pred": len(preds), "pred_raw": len(preds_raw), "gt": len(gts)},
        "pred_nearest_offset_hist": offset_hist,
        "per_gh": per_gh,
        "fp_boxes": [preds[i] for i in fp],
        "fn_boxes": [gts[i] for i in fn],
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "loop_state.json").write_text(json.dumps(report, indent=2))
    if overlays:
        _overlays_global(regions, preds, gts, tp_pairs, fp, fn)

    print(f"=== tol={tol}px  macro_F1={report['macro_f1']}  micro_F1={report['micro_f1']} "
          f"(TP={tot_tp} FP={tot_fp} FN={tot_fn}) pred={len(preds)}(raw {len(preds_raw)}) gt={len(gts)} ===")
    for gh in TRAIN_GHS:
        r = per_gh[gh]
        print(f" GH{gh:>2}: pred={r['pred']:>2} gt={r['gt']:>2} "
              f"TP={r['tp']:>2} FP={r['fp']:>2} FN={r['fn']:>2} F1={r['f1']}")
    print(f"pred nearest-offset hist (<=px): {offset_hist}")
    return report


def _overlays_global(regions, preds, gts, tp_pairs, fp, fn):
    import cv2
    page = cv2.imread(str(PAGE_IMG))
    tp_p = {pi for pi, gi, off in tp_pairs}
    for gh in TRAIN_GHS:
        r = next(x for x in regions if x["group_home"] == gh)
        y0, x0, y1, x1 = r["box_px"]
        crop = page[y0:y1, x0:x1].copy()
        for i, b in enumerate(preds):
            if not (y0 <= center(b)[0] <= y1 and x0 <= center(b)[1] <= x1):
                continue
            col = (255, 128, 0) if i in tp_p else (0, 0, 255)
            cv2.rectangle(crop, (b[1] - x0, b[0] - y0), (b[3] - x0, b[2] - y0), col, 3)
        for i in fn:
            b = gts[i]
            if not (y0 <= center(b)[0] <= y1 and x0 <= center(b)[1] <= x1):
                continue
            cv2.rectangle(crop, (b[1] - x0, b[0] - y0), (b[3] - x0, b[2] - y0), (0, 255, 255), 3)
        cv2.imwrite(str(OUTDIR / f"overlay_GH{gh}.png"), crop)


def _overlays(regions, pred_by_gh, gt_by_gh, tol):
    import cv2
    page = cv2.imread(str(PAGE_IMG))
    for gh in TRAIN_GHS:
        r = next(x for x in regions if x["group_home"] == gh)
        y0, x0, y1, x1 = r["box_px"]
        crop = page[y0:y1, x0:x1].copy()
        preds = pred_by_gh.get(gh, [])
        gts = gt_by_gh.get(gh, [])
        tp_pairs, fp, fn = match(preds, gts, tol)
        tp_p = {pi for pi, gi, off in tp_pairs}
        for i, b in enumerate(preds):
            col = (255, 128, 0) if i in tp_p else (0, 0, 255)  # TP blue, FP red
            cv2.rectangle(crop, (b[1] - x0, b[0] - y0), (b[3] - x0, b[2] - y0), col, 3)
        for i, b in enumerate(gts):
            if i in fn:
                cv2.rectangle(crop, (b[1] - x0, b[0] - y0), (b[3] - x0, b[2] - y0), (0, 255, 255), 3)  # FN yellow
        cv2.imwrite(str(OUTDIR / f"overlay_GH{gh}.png"), crop)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=int, default=2)
    ap.add_argument("--overlays", action="store_true")
    args = ap.parse_args()
    run(tol=args.tol, overlays=args.overlays)
