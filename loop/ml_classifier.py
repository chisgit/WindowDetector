"""
Option 3 — window-vs-not-window classifier on detector candidate crops.

The deterministic detector has near-perfect box geometry + high recall; its only
material error is window-vs-door/wall false positives (the macro@10=0.885 ceiling).
This trains a classifier on candidate crops to reject those FPs.

Data: page-17 corrected GT (projects/7dde299f.../). Candidates harvested per group
home from raw wall-band + final detector output, labeled by GT match (tol15).
CV: leave-one-group-home-out (4 plans) so we measure generalization across plans.

Run: PYTHONPATH=. python3 loop/ml_classifier.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.services import stantec_detector_core as det  # noqa
import loop.harness as H  # noqa

P = ROOT / "projects" / "7dde299f-ec42-4424-8f46-f865bfdb4a3b"
PDF = P / "original.pdf"
IMG = P / "page_17.png"
TRAIN_GHS = ["1", "2", "3", "4", "26"]
MATCH_TOL = 15  # a candidate is a window if within this of a GT box


def _orient(b):
    return "h" if (b[3] - b[1]) >= (b[2] - b[0]) else "v"


def harvest_candidates(regions):
    """Per GH: final detector preds + raw wall-band candidates (full-page coords),
    deduped. Returns {gh: [box,...]}."""
    page = cv2.imread(str(IMG))
    out = {}
    for gh in TRAIN_GHS:
        r = next(x for x in regions if x["group_home"] == gh)
        y0, x0, y1, x1 = r["box_px"]
        boxes = []
        # final detector output
        for w in det.detect_stantec_windows_for_page(PDF, 17, page_image_path=IMG, group_homes=[gh]):
            boxes.append(list(w["box_px"]))
        # raw wall-band candidates (more negatives: doors/walls considered)
        crop = page[y0:y1, x0:x1].copy()
        _filtered, raw = det.stantec_wall_band_windows(crop)
        for c in raw:
            boxes.append([c.y + y0, c.x + x0, c.y + c.h + y0, c.x + c.w + x0])
        out[gh] = H.dedup(boxes, tol=6)
    return out


def features(page_gray, b):
    """Feature vector for a candidate crop. Geometry + ink structure + rail/tick stats."""
    y0, x0, y1, x1 = b
    h = y1 - y0
    w = x1 - x0
    if h <= 0 or w <= 0:
        return None
    m = 6
    H_, W_ = page_gray.shape
    roi = page_gray[max(0, y0 - m):min(H_, y1 + m), max(0, x0 - m):min(W_, x1 + m)]
    if roi.size == 0:
        return None
    bw = (roi < 200).astype(np.float32)
    rh, rw = bw.shape
    long_side = max(w, h)
    short_side = min(w, h)
    vert = h >= w
    dens = bw.mean()

    def band(a, b_, axis):
        sl = bw[a:b_, :] if axis == 0 else bw[:, a:b_]
        return float(sl.mean()) if sl.size else 0.0

    t3r = max(1, rh // 3)
    t3c = max(1, rw // 3)
    top = band(0, t3r, 0); midr = band(t3r, 2 * t3r, 0); bot = band(2 * t3r, rh, 0)
    left = band(0, t3c, 1); midc = band(t3c, 2 * t3c, 1); right = band(2 * t3c, rw, 1)

    # projection peak counts (cap bands / mullions)
    rowp = bw.sum(axis=1) / max(1, rw)
    colp = bw.sum(axis=0) / max(1, rh)

    def peaks(p, thr):
        idx = np.where(p >= thr)[0]
        if len(idx) == 0:
            return 0
        n = 1
        for i in range(1, len(idx)):
            if idx[i] > idx[i - 1] + 2:
                n += 1
        return n
    row_peaks = peaks(rowp, 0.3)
    col_peaks = peaks(colp, 0.3)

    # two-rail signatures (both axes): rails on the two long edges, hollow middle
    if vert:
        rail = ((bw[:, :t3c].mean(axis=1) >= 0.25) & (bw[:, rw - t3c:].mean(axis=1) >= 0.25)
                & (bw[:, t3c:rw - t3c].mean(axis=1) <= 0.6)).mean()
    else:
        rail = ((bw[:t3r, :].mean(axis=0) >= 0.25) & (bw[rw - 0:, :].mean(axis=0) if False else bw[rh - t3r:, :].mean(axis=0) >= 0.25)
                & (bw[t3r:rh - t3r, :].mean(axis=0) <= 0.6)).mean()
    rail = float(rail)

    # symmetry
    sym_lr = 1.0 - abs(left - right) / (left + right + 1e-6)
    sym_tb = 1.0 - abs(top - bot) / (top + bot + 1e-6)

    # USER RULE 1 — end caps: a window's two rails are closed at BOTH ends by a
    # perpendicular cap stroke (the I-beam ⊏⊐). A door sill / wall line lacks them.
    # cap_score = weaker of the two end strokes, normalised to the short side.
    if vert:  # tall: caps are horizontal strokes at top/bottom
        rsum = bw.sum(axis=1); e = max(2, rh // 8)
        cap_score = min(rsum[:e].max(), rsum[-e:].max()) / float(rw + 1e-6)
    else:     # wide: caps are vertical strokes at left/right
        csum = bw.sum(axis=0); e = max(2, rw // 8)
        cap_score = min(csum[:e].max(), csum[-e:].max()) / float(rh + 1e-6)
    cap_score = float(min(cap_score, 1.5))

    # USER RULE 2 — break in the wall: a window sits in a GAP in the thick wall band.
    # If a thick SOLID wall (contiguous dark run, not a thin rail) passes through the
    # candidate, there is no break -> not a window. Measured with extra context.
    cm = 26
    if vert:
        ctx = (page_gray[max(0, y0 - 4):min(H_, y1 + 4), max(0, x0 - cm):min(W_, x1 + cm)] < 175)
        line = ctx.mean(axis=0) if ctx.size else np.zeros(1)  # per column
    else:
        ctx = (page_gray[max(0, y0 - cm):min(H_, y1 + cm), max(0, x0 - 4):min(W_, x1 + 4)] < 175)
        line = ctx.mean(axis=1) if ctx.size else np.zeros(1)  # per row
    run = mx = 0
    for v in line:
        if v >= 0.9:
            run += 1; mx = max(mx, run)
        else:
            run = 0
    wall_thick_run = float(mx)  # px of solid wall crossing the candidate (0 for a real window gap)

    return [
        float(long_side), float(short_side), float(short_side) / float(long_side + 1e-6),
        float(vert), dens, top, midr, bot, left, midc, right,
        float(row_peaks), float(col_peaks), rail, sym_lr, sym_tb,
        float(midr), float(midc),  # interior density (door leaf fills interior)
        cap_score, wall_thick_run,  # user-derived: end caps + wall-break
    ]


def build_dataset():
    regions = det.detect_stantec_plan_regions(PDF, 17, page_image_path=IMG)
    gt = H.load_ground_truth()
    page_gray = cv2.cvtColor(cv2.imread(str(IMG)), cv2.COLOR_BGR2GRAY)
    cand = harvest_candidates(regions)
    X, y, groups, boxes = [], [], [], []
    for gh in TRAIN_GHS:
        for b in cand[gh]:
            f = features(page_gray, b)
            if f is None:
                continue
            # label: window if matches any GT box (same orientation, within tol)
            lab = 0
            for g in gt:
                if _orient(b) == _orient(g) and H.corner_offset(b, g) <= MATCH_TOL:
                    lab = 1
                    break
            X.append(f); y.append(lab); groups.append(gh); boxes.append(b)
    return np.array(X), np.array(y), np.array(groups), boxes


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneGroupOut
    X, y, groups, boxes = build_dataset()
    print(f"dataset: {len(y)} candidates  positives(window)={int(y.sum())}  negatives={int((1-y).sum())}")
    for gh in TRAIN_GHS:
        msk = groups == gh
        print(f"  GH{gh}: {msk.sum()} cands, {int(y[msk].sum())} windows")
    logo = LeaveOneGroupOut()
    oof = np.full(len(y), -1.0)
    for tr, te in logo.split(X, y, groups):
        clf = RandomForestClassifier(n_estimators=400, max_depth=6, class_weight="balanced", random_state=0)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    # Post-filter framing: REJECT candidate only if P(window) < thr. We want to drop
    # negatives (FPs) while keeping (near) all positives (real windows).
    print("thr  | windows_kept  negatives_rejected")
    for thr in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
        kept_pos = int(((oof >= thr) & (y == 1)).sum())
        rej_neg = int(((oof < thr) & (y == 0)).sum())
        print(f"{thr:4.2f} | {kept_pos:2d}/{int(y.sum())} windows   {rej_neg:2d}/{int((1-y).sum())} negatives rejected")
    # operating point that keeps >= 98% of windows
    import numpy as _np
    pos_p = _np.sort(oof[y == 1])
    thr98 = pos_p[max(0, int(0.02 * len(pos_p)) - 1)] if len(pos_p) else 0
    print(f"\nthreshold keeping ~98% windows: {thr98:.3f}")
    kept = int(((oof >= thr98) & (y == 1)).sum()); rej = int(((oof < thr98) & (y == 0)).sum())
    print(f"  -> keeps {kept}/{int(y.sum())} windows, rejects {rej}/{int((1-y).sum())} negatives")

    # DECISIVE TEST: apply post-filter (out-of-fold prob) to the FINAL detector
    # output and re-score F1@10 vs GT. Does it remove the window-vs-door FPs?
    print("\n=== post-filter applied to FINAL detector output (F1@tol10) ===")
    prob_of = {tuple(b): p for b, p in zip(boxes, oof)}
    gt = H.load_ground_truth()
    for thr in (0.10, 0.15, 0.18, 0.20):
        kept_preds = []
        for gh in TRAIN_GHS:
            for w in det.detect_stantec_windows_for_page(PDF, 17, page_image_path=IMG, group_homes=[gh]):
                b = tuple(w["box_px"])
                # nearest harvested candidate prob (final preds are in the harvest)
                p = prob_of.get(b)
                if p is None:
                    # match by closeness
                    best = min(boxes, key=lambda c: H.corner_offset(b, c))
                    p = prob_of[tuple(best)] if H.corner_offset(b, list(best)) <= 8 else 1.0
                if p >= thr:
                    kept_preds.append(list(b))
        kept_preds = H.dedup(kept_preds)
        tp, fp, fn = H.match(kept_preds, gt, 10)
        print(f"  thr={thr:.2f}: F1@10={H.f1(len(tp),len(fp),len(fn)):.4f} TP={len(tp)} FP={len(fp)} FN={len(fn)} (preds={len(kept_preds)})  [baseline 0.887: TP51 FP6 FN7]")

    # ARCHITECTURE TEST: high-recall proposer (ALL harvested candidates incl. raw)
    # + classifier filter (out-of-fold prob). Can recover FNs AND remove FPs.
    print("\n=== high-recall proposer + classifier filter (F1@tol10) ===")
    boxes_arr = boxes
    for thr in (0.10, 0.15, 0.20, 0.30, 0.40):
        dets = [boxes_arr[i] for i in range(len(boxes_arr)) if oof[i] >= thr]
        dets = H.dedup(dets)
        tp, fp, fn = H.match(dets, gt, 10)
        print(f"  thr={thr:.2f}: F1@10={H.f1(len(tp),len(fp),len(fn)):.4f} TP={len(tp)} FP={len(fp)} FN={len(fn)} (dets={len(dets)})")
