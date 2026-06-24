# Session Handoff — Window Detection Accuracy Loop

Last updated: 2026-06-24. Read this first when resuming.

## Current state
- **Accuracy (page 17, all 6 group homes, GT=86 windows): macro F1@10 = 0.926, micro = 0.936, FP=5, FN=6.**
- Pipeline = deterministic detector + door-swing reject + **ML post-filter** (RandomForest, reject if P(window) < 0.30).
- Branch: `window-detector-accuracy-loop` (latest commit pushed). Safe restore: branch `88pct-deterministic-baseline` / tag `loop-checkpoint-0.885`.
- GitHub remote: `chisgit/WindowDetector`.

## How to run (local, deterministic — no Gemini/cloud)
Python env: a uv venv at `/home/user/wd-venv` (the repo `.venv` is a locked Windows venv, unusable from WSL).
If it's gone, recreate:
```
uv venv /home/user/wd-venv --python 3.12
uv pip install --python /home/user/wd-venv fastapi uvicorn pymupdf python-dotenv pillow opencv-python-headless pydantic python-multipart numpy scikit-learn joblib
```

**Start the server (ML filter on):**
```
cd /mnt/c/Users/User/WindowDetector
WINDOW_ML_FILTER=1 PYTHONPATH=. /home/user/wd-venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
→ UI at http://localhost:8000 . The Gemini import is optional (AI button returns 503; deterministic model works). Stop with `pkill -f uvicorn`.

**Score / measure accuracy:**
```
WINDOW_ML_FILTER=1 PYTHONPATH=. /home/user/wd-venv/bin/python loop/harness.py --tol 10
```
(omit `WINDOW_ML_FILTER=1` for the deterministic-only baseline; add `WINDOW_DOOR_FILTER=0` to disable the door reject.)

**Retrain the classifier after new corrections:**
```
PYTHONPATH=. /home/user/wd-venv/bin/python loop/train_classifier.py   # -> loop/window_clf.joblib (thr 0.30)
```
Restart the server afterwards to load the new model.

## The correction workflow (the proven lever to higher accuracy)
1. Open a plan/page in the UI, let the detector run.
2. Correct it: **real windows keep their `W-NN` label; rename any wrong box with a reason** (e.g. `DOOR`, `STAIRS`, `no caps`, `not in a wall`, `washer dryer`). The harness now treats **any non-`W-NN` label as a hard negative** (`loop/harness.py:_is_reject_label`).
3. Retrain (above), re-measure, commit.
Corrections live in `projects/7dde299f-…/correction_log.jsonl` (last `page_windows_saved` per page = ground truth).

## Key decisions already made (don't re-litigate)
- **Deterministic hard rules for window-vs-not DON'T work here.** Tested caps, in-wall, crisscross, mullion, two-rail, thick-wall, Hough-arc — every signal OVERLAPS (a non-window can cap-score higher than every real window). They only work *in combination* = the ML classifier. The user's rules (caps, wall-break) are encoded as ML **features** (`cap_score`, `wall_thick_run` in `loop/ml_classifier.py:features`). Adding in-wall+diag as extra features regressed CV → reverted. See LOOP_LEARNINGS.md.
- **The one deterministic reject that works:** door-swing arc (`is_door_swing_candidate` in `stantec_detector_core.py`), but only for *over-extended* door-spans (>165px); compact door arcs are left to the ML.
- **Threshold 0.30** chosen to favor recall (user tolerance ~3 errors/detection) and generalize to a new file. It's tuned to corrected data; a brand-new uncorrected plan will be rougher until corrected once.
- **Page 18 = construction sheets of the same homes**, but drawn at a different scale+position per home + extra noise, so page-17 answers do NOT auto-transfer (ORB/ECC/median/ink-overlap all fail on GH2/4/5/26; GH1/GH3 register and the detector scores ~0.87 there). The detector DOES generalize to page 18; GH1/GH2 were corrected manually.

## What's next
- User is moving to a **brand-new PDF** (real generalization test). Run the same loop: detect → correct in UI → retrain → measure. Expect a rougher first pass (ML has only seen this one PDF's 6 plans); corrections fix it fast.

## Files
- `app/services/stantec_detector_core.py` — detector (single source of truth). Door-swing reject + ML hook (`WINDOW_ML_FILTER`) + multi-pane merge.
- `loop/harness.py` — scoring + GT/negatives loading (`load_ground_truth`, `load_hard_negatives`, `_is_reject_label`).
- `loop/ml_classifier.py` — features + leave-one-group-home-out CV.
- `loop/train_classifier.py` — trains/saves `loop/window_clf.joblib`.
- `loop/ml_filter.py` — inference-time post-filter used by the detector.
- `docs/solutions/LOOP_LEARNINGS.md` — full iteration log + every reverted attempt with reasons.
