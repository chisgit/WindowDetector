"""
Optional ML post-filter for detector output (Option 3).

Loads the trained window-vs-not classifier and rejects detector candidates the
model is confident are NOT windows (P(window) < reject_threshold), keeping near-100%
of real windows. Safe by design: high-recall operating point; if the model or
sklearn/joblib is unavailable it is a no-op.

Enable in the detector by setting env WINDOW_ML_FILTER=1 (default off, so the
deterministic baseline is never altered unless explicitly requested).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

_MODEL_PATH = Path(__file__).resolve().parent / "window_clf.joblib"
_cache = {"loaded": False, "bundle": None}


def _load():
    if _cache["loaded"]:
        return _cache["bundle"]
    _cache["loaded"] = True
    try:
        import joblib
        _cache["bundle"] = joblib.load(_MODEL_PATH) if _MODEL_PATH.exists() else None
    except Exception:
        _cache["bundle"] = None
    return _cache["bundle"]


def filter_windows(page_gray: np.ndarray, windows: list[dict]) -> list[dict]:
    """windows: list of {'box_px':[ymin,xmin,ymax,xmax], ...}. Returns kept subset."""
    bundle = _load()
    if bundle is None or not windows:
        return windows
    from loop.ml_classifier import features
    clf = bundle["model"]
    thr = bundle.get("reject_threshold", 0.18)
    kept = []
    for w in windows:
        f = features(page_gray, w["box_px"])
        if f is None:
            kept.append(w); continue
        p = clf.predict_proba(np.array([f]))[0, 1]
        if p >= thr:
            kept.append(w)
    return kept
