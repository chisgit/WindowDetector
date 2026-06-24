# PLAN — Supporting a New Window Symbol Family (Barrie set)

Status: **proposed / not started**. Created while wiring the Barrie PDF into the
deterministic-Stantec pipeline. Read alongside `docs/solutions/SESSION_HANDOFF.md`
and `CLAUDE.md` ("Expect multiple window symbol families over time. Keep detector
logic modular").

## Why this plan exists

The `deterministic-stantec` backend was tuned on the Stantec group-home plans
(page-17 set). The Barrie PDF (`Barrie-Attachment 1 - Drawings.pdf`) is a
different practice's drawing set:

- It does **not** carry the Stantec "Group Home N" title blocks, so
  `is_stantec_pdf()` returns `False` and `find_stantec_plan_titles()` finds 0
  title regions. Plan-region detection therefore falls back to the generic
  `detect_floor_plan_regions()` (contour-based), which still works.
- We have now **removed the legacy-local fallback** for this backend (see
  `app/main.py`, `run_local_region_detection`) so the Stantec wall-band window
  detector runs on Barrie regions directly. It returns boxes, but their
  correctness on this layout is **unvalidated**.

Open question this plan answers: *if Barrie draws windows with a different symbol
than the Stantec set, how do we add that family without breaking the existing
detector?*

## Step 0 — Confirm whether Barrie windows are actually a new family (do FIRST)

Do not build anything until this is answered with evidence:

1. Pick 2–3 **real** Barrie floor-plan pages (NOT page 1 — page 1 is a cover /
   "List of Drawings" sheet; its detections are expected to be junk).
2. Run `deterministic-stantec` per plan region in the UI.
3. Correct one page in the UI (real windows keep `W-NN`; wrong boxes renamed
   with a reason — same convention as the page-17 loop).
4. Compare the corrected windows' symbol geometry to the Stantec family:
   wall-band thickness, cap stroke length/spacing, pane/mullion pattern,
   horizontal-run merging. Capture crops in `scratch/barrie_*`.

**If the existing wall-band features already fire on Barrie windows** → it is NOT
a new family; just feed corrections into the existing ML post-filter
(`loop/train_classifier.py`) and stop. Cheapest path; prefer it.

**Only if the symbol is geometrically distinct** → proceed below.

## Step 1 — Make symbol families first-class (modular registry)

Today the detector is one monolithic crop pipeline (`_run_stantec_crop` in
`app/services/stantec_detector_core.py`). Refactor toward a small registry so
families are additive, not edits to shared code:

- Define a `WindowFamily` protocol: `candidates(crop, text_mask) -> list[Candidate]`
  plus a `name` and a cheap `applies(crop)` gate.
- Wrap the current Stantec logic as `family="stantec_wall_band"` (no behavior
  change — this must be a pure extraction, verified by the harness).
- New families (e.g. `barrie_*`) implement the same protocol and register
  themselves. The crop runner unions candidates across applicable families, then
  runs the shared post-steps (overlap suppression, door-swing reject, ML filter).

Guardrail: families must be **opt-in per detection / PDF profile** so adding a
Barrie family cannot regress the page-17 macro F1 (currently 0.926 @10).

## Step 2 — Implement the Barrie family

- Encode the distinct symbol geometry from Step 0 as detector primitives /
  candidate generators, mirroring how `cap_score` / `wall_thick_run` are encoded.
- Add Barrie crops to the ML feature/training set; retrain; measure with a
  Barrie-specific harness ground-truth (new fixture, do not pollute page-17 GT).

## Step 3 — Regression gates before merge

- `loop/harness.py --tol 10` on the page-17 set must stay **≥ 0.926 macro F1**.
- New Barrie harness fixture reports its own F1.
- Both gates green → merge.

## Things to decide later / flag for the user

- **Force-Stantec-on-non-Stantec is now the default for this backend.** Good for
  experimentation, but means a future non-Stantec PDF will no longer silently
  use legacy-local. If that surprises anyone, reintroduce a UI toggle instead of
  the automatic gate.
- **Page-1-type cover sheets** produce false windows. Consider gating detection
  to pages classified `is_floor_plan=True` AND excluding "List of Drawings"
  cover sheets.
- The locked Windows `.venv` in the repo root is unusable from WSL and was the
  original source of the `cannot identify image file` crash. Path handling is now
  hardened (absolute `PROJECTS_DIR` + backslash normalization), but consider
  deleting/ignoring that `.venv` so nobody runs against it again.
