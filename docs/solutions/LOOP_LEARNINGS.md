# Window-Detection Accuracy Loop — Learnings

Spec: `WINDOW_98_LOOP.json`. Harness: `loop/harness.py`. State: `loop/out/loop_state.json`.
Detector (single source of truth): `app/services/stantec_detector_core.py`.
Ground truth: last `page_windows_saved` in `projects/7dde299f.../correction_log.jsonl` (58 boxes, page 17 @ 9362×6622).
Train/score set: GH1, GH2, GH3, GH26. GH4/GH5 = inference handoff only.

Metric: TP if same orientation AND all 4 box edges within `tol` px of a GT box (length+placement). F1 per plan.

## Run env note
WSL default python3 (3.12) lacks cv2/fitz; the repo `.venv` is a Windows venv (unusable from WSL).
Installed `opencv-python-headless` + `PyMuPDF` into user site via `pip install --user --break-system-packages`.
Run harness with: `PYTHONPATH=. python3 loop/harness.py --tol N [--overlays]`.

## Baseline (iteration 0, unmodified detector)
GT distribution over plans: GH1=15, GH2=16, GH3=14, GH26=13 (=58).

| tol px | macro F1 | micro F1 | TP | FP | FN |
|-------:|---------:|---------:|---:|---:|---:|
| 2  | 0.639 | 0.638 | 37 | 21 | 21 |
| 5  | 0.777 | 0.776 | 45 | 13 | 13 |
| 10 | 0.808 | 0.810 | 47 | 11 | 11 |
| 20 | 0.808 | 0.810 | 47 | 11 | 11 |

Per-plan @tol=10: GH1 F1=0.867, GH2 F1=0.727, GH3 F1=1.000, GH26 F1=0.640.

### KEY INSIGHT — two separate problems
1. **Box drift (~10 windows):** correctly-detected windows whose boxes sit 3–10px from GT.
   They pass at tol≥5 but double-count (FP+FN) at tol=2. This is cap/pane-trim precision.
2. **Genuine errors (11 FP + 11 FN, persist even at tol=20):** real misdetections.
   These cap detection quality at F1≈0.81 regardless of tolerance. **Primary target.**

### CAUTION on the ±2px target
The ground truth boxes are **hand-drawn in the UI** (corrected_windows have no detector fields).
Human box placement on a 9362px-wide render carries its own several-px noise, so demanding
detector↔GT agreement within ±2px is partly measuring GT drawing noise, not detector error.
Recommend treating **tol≈10 F1 as the true "found the right windows" score** and using ±2px only
as a secondary precision gauge. Flag for user decision.

## Genuine error patterns (from tol=10 FP/FN, visually confirmed via loop/out/zoom_*.png)
- **P1 thin-wall vertical caps missed** (GH26 right wall x4563: 2 stacked V windows, double-rail+end-ticks
  on a thin single-line exterior wall). Wall-band detection wants a thick band → skips thin-wall caps.
- **P2 dimension-line-crossed horizontal cap missed** (GH26 H@4057 x4198: dimension line+arrowhead
  crosses the cap). Matches CLAUDE.md rule: recover underlying cap/pane evidence under dimension lines.
- **P3 vertical extent not trimmed to cap ink** (GH2 V@~1993 x3561 etc.): box extends down into wall/door
  space below the cap; shifted ~30px. Several verticals affected.
- **P4 horizontal two-pane under-merge** (GH1 y2240 x928: only left pane captured; GT spans both panes).
- **P5 horizontal over-extension** (GH2 226px vs GT 162px): horizontal cap extended too far along wall.
- **P6 pure FP long horizontals** (205–241px, no GT nearby): likely dimension/wall lines counted as windows.

## Iteration log
### Iter 0 — baseline. No code changes. macro_F1@2=0.639, @10=0.808 (per-crop scoring).

### Iter 1 — harness bug fix (measurement, not detector)
Per-crop scoring double-counted windows in the GH2/GH26 crop-overlap band as FP in one
plan and FN in the other. Switched harness to GLOBAL page-level scoring: union of
predictions across trained crops, deduped (`dedup`, 6px), matched vs all 58 GT.
Result @10: 0.808 -> 0.832 (FP/FN 11/11 -> 9/10). No detector change.

### Iter 2 — generic two-rail span trim for thin horizontal caps  ✅ KEPT
Pattern P5/P6 (horizontal over-extension): `detect_generic_interior_thin_horizontal_caps`
uses a (46,3) morphology that merges a window cap with an adjacent solid wall pier / wall
run into one 75–230px component; `thin_cap_preserve` then kept it full-width (no trim).
Fix: new `trim_thin_cap_to_window_span()` — trims a horizontal cap to the longest
contiguous span carrying the two-rail signature (rail ink top+bottom, hollow middle),
dropping solid-pier (filled middle) and empty-wall tails. Applied in the thin_cap_preserve
branch. Bridges short gaps (<=14px = mullion between panes of ONE window) but not wide gaps
(pier/wall), so two/three-pane windows stay whole. Caps already two-rail full-width are
returned unchanged -> correctly-sized caps never shortened.
Before@10 macro=0.832 -> After@10 macro=0.863 (TP 48->50, FP 9->7, FN 10->8).
@5: 0.800->0.847. @2: 0.647 (flat). Per-GH@10: GH1 0.87->0.93, GH2 0.73->0.82,
GH3 1.0 (no regression), GH26 0.64->0.70. graphify updated.

### Iter 4 — keep full-height tall vertical windows  ✅ KEPT
Pattern: vertical under-height (GH26 x2553 V@3827: detected 91px, GT 162px).
`promote_upper_component_and_reject_wall_space` replaced the candidate with an upper
component capped at min(92,..) height, assuming "cap above blank wall space below". For a
genuinely tall continuous window this truncates it. Fix: before truncating, if the full
raw candidate has continuous cap-band evidence (cap_band_count>=4) and r.h<=175, keep its
full height ("+full_component_keep") instead of the 92px upper pane.
@10 macro 0.863 -> 0.885 (TP 50->51, FP 7->6, FN 8->7). @5/@2 unchanged. No regressions.

## Current status @tol=10: macro_F1=0.885, micro_F1=0.887 (TP=51 FP=6 FN=7)
Remaining errors (next targets):
- two_rail_recovery FP spans (H205@y780, H160@y4801) — different code path (~L1425), not the
  thin-cap branch. (Iter-3 attempt to trim ALL horizontals removed -> see mistake log.)
- vertical placement drift / under-height (GH2 V@2022,V@2342; GH26 V@3827) — cap-ink align.
- thin-wall vertical misses (GH26 x4563 V@4259,V@4381) + stacked 2nd (x2553 V@4402).
- dimension-line-crossed horizontal miss (GH26 H@4057 x4198).
- GH1 two-pane (y2240 x928) still 1 FP/1 FN.

## Mistake log
- Iter-3 (REVERTED): applied `trim_thin_cap_to_window_span` to ALL horizontals in the final
  expand loop to also catch two_rail_recovery FPs. It shaved a few px off already-correct
  windows (tol=5 0.847->0.830, tol=2 0.647->0.632) WITHOUT removing the target FPs (they
  don't pass through that loop). Net negative -> reverted. Lesson: scope the trim to the
  path that produces the over-extension; don't blanket-apply geometry edits to good boxes.
