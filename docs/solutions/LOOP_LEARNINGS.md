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
- Iter-5 (REVERTED): scoped `trim_thin_cap_to_window_span` to the two_rail_recovery output
  in recover_compact_horizontal_cap_windows. Did NOT remove the target FPs and regressed
  tol=5 (0.847->0.830). Root cause: a window->wall->door span is two-railed along its WHOLE
  width (the two wall lines mimic two rails), so the two-rail/hollow-middle test cannot
  distinguish window from wall. NEXT: these two_rail FPs (H205@y780, H160@y4801, H191@y2579)
  need a MULLION/PANE-TICK discriminator -- a real multi-span window has internal vertical
  mullion ticks at pane boundaries; the wall/door gap between a window and a door has none.
  Trim the two-rail span to the run bounded by mullion ticks.

## Iter-6/7/8 attempts (all REVERTED — drift recorded)
- Iter-6 mullion-span discriminator for two_rail FPs: drift -0.014 macro@10, REGRESSED 1 real
  window (H81@y3732x3621 -> FN) and removed 0 of the 3 target FPs. Door frames / wall piers
  are vertical strokes = false stiles (wall not rejected); some real windows lack a detectable
  mid-gap stile (false rejection). Signal not separable -> dropped per "helps few/hurts" rule.
- Iter-7 lower expand guard 90->72 (to merge GH1 single pane into two-pane): NO-OP (0 change).
  expand_horizontal_to_matching_pane_run's internal >=3-pane-stile / pane_w / density checks
  do not fire for this window, so the guard was never the blocker. Reverted (no benefit).

## Deeper diagnoses (for whoever resumes)
- GH1 two-pane (H79 vs H158, src two_rail_recovery): only the left 79px pane is detected; the
  two-pane MERGE never happens. expand internals don't find the neighbour pane run. Needs work
  inside refine_horizontal_two_pane_from_raw / has_two_horizontal_panes for this geometry.
- Stacked-2nd vertical (GH26 x2553 V@4402, 155px): FILTERING miss, NOT detection. Raw HAS it
  (derived_from_vertical_wall_band+pane_merge, x376 y1975 w14 h156 in GH26-crop coords) but
  stantec_wall_band_windows drops it while keeping its identical twin above (V@3827, fixed in
  Iter4). Recover by understanding why wall_band filters one of two identical stacked caps.
- Thin-wall verticals x4563 (V@4259, V@4381): CROP-COVERAGE miss. (4563,4259) lies OUTSIDE
  every title-based crop (GH26 x<=4204; GH3 y<=3387) -> never fed to the detector. Fix is in
  stantec_crop_from_title (widen/overlap crops) -> structural, affects all plans.

## Iter-9 deep dive: stacked-2nd vertical — WHY it resists a clean fix (no code change)
Traced the lost lower window (GH26 x2553 V@4402) to step 7 (promote_upper_component_and_
reject_wall_space). Both stacked windows are geometrically identical real windows, but:
- cap_band_count (horizontal bands) returns 1 for BOTH within their true width — vertical
  rails span full height = one band. The UPPER only survives because adjacent dimension ticks
  in the ±6px ROI padding inflate its count to 9 (a fluke), passing bands>=3. The lower lacks
  adjacent ticks -> count 1 -> rejected by `h>=88 and bands<3`.
- Tried vertical two-rail discriminator (vtr = left&right rail cols + hollow middle): LOWER
  real window = 0.91 (good), but UPPER real window = 0.026 (fails) because candidate boxes
  don't tightly bracket the rails, so the right-third test misses. Alignment-sensitive.
CONCLUSION: 3 discriminators tried (mullion-span, cap_band_count, vtr); each fails on >=1 real
window. The boxes reaching promote are not rail-aligned, so box-level structure tests are
unreliable. A robust fix needs to FIRST snap vertical candidates to their rail structure
(refactor), THEN test — beyond the safe per-rule gated approach. Left for a deeper session.

## Iter-10 (REVERTED): vertical_rail_signature as OR-acceptance in promote
Added a vtr (left+right rail cols, hollow middle) acceptance path to promote's full-keep and
band-count rejection. Drift: -0.025 macro@10 (0.885->0.860). Regressed GH1 x1052 (correct 82px
window OVER-promoted to 171px via full_component_keep+vtr, grabbing wall space) and only
half-recovered the stacked-2nd (un-rejected as a wrong-size 89px box, not full 155).
ROOT CAUSE (confirms plateau): patching at the promote stage is too late -- candidate boxes are
not rail-aligned, so one vtr threshold both recovers one window and over-grows another. The
correct fix is to RAIL-SNAP vertical candidates at generation/trim time (trim_candidate_to_cap_ink
vertical branch), THEN the downstream tests are reliable. That is a refactor of a function run on
ALL verticals = real regression risk = needs a dedicated, well-gated session.

## Iter-11 (REVERTED to tag loop-checkpoint-0.885): the rail-align refactor
Made the coherent refactor: trust wall-band-validated verticals and relax the 3 coupled gates
that reject tall narrow rail-windows -- (1) L350 height ceiling 125->175, (2) trim L595
band-count rejection skipped when 'vertical_wall_band' in source, (3) promote cap_band_count
rejection skipped likewise. Gate result: macro@10 0.885 -> 0.811 (-0.074); GH3 BROKE 1.0->0.867;
pred 57->61 (+4 FP across plans); tol2 0.647->0.558.
DEFINITIVE CONCLUSION: the 3 gates are NOT redundant. wall_band's vertical candidate list
genuinely contains doorway / wall-space FPs that those gates correctly remove. The SAME gates
also reject ~1-2 real tall-narrow windows, but window-vs-door is not separable at this signal
level (mullion-span, vtr, cap_band_count, and rail-snap ALL fail on >=1 real window). Reverted
to tag. The last mile (0.885 -> 0.98) needs a fundamentally better signal, not another local
rule: e.g. higher-DPI render to expose cap-tick geometry, threading exact cap y-positions
through a redesigned Candidate, or a small learned classifier on cap crops.

## Option 1 (higher-DPI) VERDICT: not worth the scale-aware refactor
- Supersampling (render 400 -> downscale to 200 reference): NO gain (@10 identical 0.887;
  @5/@2 slightly worse from edge shift). So cleaner input is not the bottleneck.
- True 400-DPI crops DO show clearer mullion-tick structure visually, BUT a high-DPI tick
  detector still FAILS to separate window from wall/door: FP over-extensions (door swings,
  wall junctions, dimension lines) register as MORE tick-groups than a real two-pane window
  (FP_H205 groups=5, FP_H191=3, real 2-pane=4). Hand-crafted features get fooled at any DPI.
- Conclusion: resolution is not the limiter; the subtle window-vs-door distinction is. The
  detector's pixel constants are 200-DPI-tuned, so a true-400 run also needs a large scale-aware
  refactor for uncertain gain. NOT WORTH IT. The real lever is a learned classifier (Option 3)
  trained on corrected crops across MANY drawings (the UI correction workflow generates that data).

## GH4 corrections — caps + wall-break as ML FEATURES (not hard rules)
User corrected GH4 in the UI, relabelling 4 detections with the REASON they are not windows:
"no caps", "No break in the wall" (x2), "Not window". (Harness now treats any reject-labelled
box as a hard NEGATIVE, not GT: see _is_reject_label / load_hard_negatives.) GH4 detector recall
is perfect (15/15); the only error is those 4 FPs.
- Tried both reasons as DETERMINISTIC hard filters first (regression measured, per project rule):
  cap-presence rule removed 4/66 real TPs; thick-wall rule removed 5/66 TPs. Same lesson as the
  earlier plateau: the 4 FPs sit in local contexts almost identical to real windows, so no single
  pixel-rule separates them without killing real windows. NOT integrated.
- Instead encoded the user's two rules as CONTINUOUS FEATURES in ml_classifier.features():
  cap_score (weaker of the two end-cap strokes / short side) and wall_thick_run (px of contiguous
  solid wall crossing the candidate, measured with 26px context). The RF then weights them with the
  other 18 features. LOGO CV: 3 of 4 FPs drop below all real windows (P 0.19/0.33/0.40 vs window
  min 0.52); FP2 (door threshold, user: "looks like a window but no caps") stays 0.65 — genuinely
  ambiguous. Operating point P<0.30 keeps 70/70 windows in CV.
- Result @10 (GT now 73, incl GH4): deterministic 0.889 -> +ML 0.919 (in-sample re-run; CV 0.904).
  GH4 0.882->0.968 (4 FP -> 1), GH1/GH3 unchanged (no regression). FP total 9 -> 4. Model retrained
  on the GH4-augmented data; reject_threshold 0.30. Lesson: turn user pixel-rules into FEATURES, not
  filters, when the classes overlap — safe (no TP loss) and lets evidence combine.

## Door-swing disqualification (user insight) — KEPT, deterministic, on by default
A door = wall OPENING (gap) + straight leaf + swing ARC (curve); windows are axis-aligned only
(no arcs). is_door_swing_candidate(): subtract H/V strokes -> residual holds leaf+arc; circle-fit
each residual component; accept as door arc when resid/radius<0.025, radius 40-115px (door-leaf
length), arc span>~57deg. Arc centre = hinge; if it falls inside an OVER-EXTENDED candidate
(long side >165px, longer than any real window) the box spans a door opening -> drop it.
- Why the over-extension guard: windows and doors are frequently ADJACENT on the same wall, so a
  naive "arc within pad" disqualifier false-fired on 7/51 real windows (a door hinge sits right at
  a neighbouring window). Requiring the candidate to be over-long protects compact real windows;
  it then targets only door-spanning over-extensions (e.g. H205). Verified 0/51 TP removed.
- Result @10: 0.885 -> 0.895 (FP 6->5), GH3 stays 1.0, no regressions. Stacks with ML: door+ML = 0.907.
- Rejected approaches (logged): HoughCircles (24 false circles on a hatched window, missed doors);
  high-DPI tick counting (door swings/junctions mimic ticks). Circle-fit on axis-subtracted residual
  is the robust arc detector. On by default; disable with env WINDOW_DOOR_FILTER=0.

## Option 2 (cap-geometry plumbing) VERDICT: not worth it — precision is already near-perfect
Measured corner-offset of the 51 matched TP windows vs hand-drawn GT:
  BEFORE any snap: median=0.0px, mean=1.3px, 37/51 within <=2px, 49/51 within <=5px.
  AFTER snapping boxes to cap-ink bbox: WORSE (median 5px) -- ink bbox grabs wall lines.
So the detector's box geometry is already essentially optimal (median 0px!). The tol2 score
(0.647) gap is NOT detector drift; it is hand-drawn-GT noise -- the ~14 windows that miss tol2
are only 3-5px off (human drawing precision on a 9362px render). Geometry plumbing has no
headroom. REFRAME: the detector is geometrically excellent + high recall; the ENTIRE remaining
error is a binary window-vs-door discrimination on ~6-7 candidate regions (the tol10 ceiling).
That is the only real lever -> Option 3 (learned classifier on candidate crops).

## Option 3 (ML candidate classifier) — WORKS, +0.018 cross-validated, opt-in
RandomForest on engineered features of detector candidate crops (geometry, ink-band stats,
two-rail signature, symmetry, mullion/peak counts). Data: 143 page-17 candidates (55 windows /
88 negatives, harvested = final preds + raw wall-band). Labeled window iff matches GT @tol15.
- Leave-one-group-home-out CV (honest generalization across plans): as a POST-FILTER rejecting
  only low-confidence candidates (P(window) < 0.18, high-recall point keeps 55/55 windows),
  F1@10 0.887 -> 0.903 (removes 2-3 of 6 FPs, keeps all TPs).
- High-recall-proposer + classifier architecture (use raw pool as proposals): WORSE (~0.73) --
  raw FPs too numerous/unseparable at this data scale. So post-filter on FINAL output is the design.
- Integration: opt-in via env WINDOW_ML_FILTER=1 (default OFF -> deterministic baseline 0.885
  unchanged; verified). ON: 0.907 macro / 0.911 micro on page 17 (slightly optimistic, model
  trained on this page). Files: loop/ml_classifier.py (features+CV), loop/train_classifier.py
  (saves loop/window_clf.joblib), loop/ml_filter.py (inference). Needs sklearn+joblib (no-op if absent).
- LIMITS: only ~half the FPs are separable, and FNs (7 missed windows) are untouched (post-filter
  can't recover non-proposed windows). Trained on ONE page -> will improve as GH4/5 + more pages
  are corrected via the UI (re-run train_classifier.py). This is the architecture that climbs to 98%.

## Page-18 generalization test
Page 18 is the CONSTRUCTION PLAN (page 17 = DEMO FLOOR PLAN) of the SAME group homes -- a
DIFFERENT drawing type, laid out at different page positions. So the page-17 GT does NOT
transfer (TP=0 even crop-relative; crops are title-relative and the layout/annotations differ).
Finding: the detector RUNS and produces plausible exterior-wall window detections on the
construction plans; GH1 count matches page-17 exactly (15). Counts: GH1=15, GH2=20, GH3=22,
GH4=33, GH5=33, GH26=24 (higher on construction sheets = denser linework -> more FP candidates,
which is exactly what the ML post-filter + per-drawing-type correction will handle). Outputs in
loop/page18_output/. Precise page-18 accuracy needs page-18 GT (correct it in the UI) -> same
data pipeline. Validates the detector handles a new drawing TYPE, the stated multi-drawing goal.

## Page-18 GH1/GH2 correction pass -- triple-pane recovery KEPT
Latest page-18 correction event: 2026-06-24T07:37:34Z, 39 annotations = 30 positives +
9 hard negatives across GH1/GH2. User correction W14/GH1 showed a valid triple horizontal
window: the detector kept only the left compact pane ([2367,1289,2390,1354]) while GT spans
all panes ([2367,1289,2390,1532]).
- Root cause: raw candidates already contained the compact pane plus adjacent two-pane cap, but
  multi-pane recovery was blocked by broad door-jamb context and by duplicate suppression against
  the narrower same-row cap.
- Fix: recover adjacent compact+wide horizontal raw runs when the union has two-rail evidence,
  allow it to replace contained narrower same-row candidates, and reject starts-inside overlaps
  so GH2 partial over-merges are not introduced.
- Page18 GH1/GH2 score @tol10: 0.7077 -> 0.7385 (TP 23->24, FP 12->11, FN 7->6).
  The recovered GH1 triple is [2367,1289,2390,1529], within 5px of GT. Page17 harness unchanged:
  @tol10 macro_F1=0.8755, micro_F1=0.884, pred=95(raw96), GT=86.

## PLATEAU: macro@10 = 0.885 is the clean, zero-regression ceiling for generic local rules.
Reaching ~0.93 requires one of (all higher-risk, need explicit go-ahead):
  (a) Rail-alignment refactor for vertical candidates (fixes stacked-2nd, drift, under-height).
  (b) Crop-coverage fix in stantec_crop_from_title (recovers x4563 windows; affects all plans).
  (c) Door-opening-aware rejection for the two_rail horizontal FPs (H205/H191/H160).
  (d) Two-pane merge rework in refine_horizontal_two_pane_from_raw (GH1).
DROP per user guidance: dimension-line-crossed horizontal (helps 1, risks many).

## Remaining targets after Iter 4 (macro@10=0.885, FP6/FN7), hardest-last
- two_rail_recovery FP spans (H205,H160,H191): need mullion-tick discriminator (see Iter-5).
- thin-wall vertical MISSES x4563 V@4259/V@4381, x2553 stacked-2nd V@4402: NEW recovery path
  for vertical caps on thin single-line exterior walls (current wall-band path needs a thick
  band). Additive -> watch for new FPs across plans.
- GH1 two-pane y2240 x928: detected left pane only; needs horizontal two-pane merge for this
  geometry (panes present but not merged by refine_horizontal_two_pane_from_raw here).
- vertical drift V@2022 x3566 (cap above, box placed ~29px low): recovery in
  recover_compact_vertical_service_caps places box too low; trim pad too small to snap up.
- bay/alcove window Vdrift x2734: FP on wall corner + missed recessed bay window (different).
- dimension-line-crossed horizontal miss H@4057 x4198 (CLAUDE.md dimension-line rule).
