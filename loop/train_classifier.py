"""
Train + save the window-vs-not-window post-filter classifier.

Trains on all corrected page data currently available (page 17). As more pages are
corrected via the UI, re-run this to improve the model. Honest generalization
estimate is the leave-one-group-home-out CV in loop/ml_classifier.py (~0.90 F1@10);
training on all data and scoring the same page is optimistic.

Run: PYTHONPATH=. python3 loop/train_classifier.py
Output: loop/window_clf.joblib
"""
import sys
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from loop.ml_classifier import build_dataset  # noqa

MODEL = ROOT / "loop" / "window_clf.joblib"


def main():
    X, y, groups, boxes = build_dataset()
    clf = RandomForestClassifier(n_estimators=400, max_depth=6,
                                 class_weight="balanced", random_state=0)
    clf.fit(X, y)
    joblib.dump({"model": clf, "reject_threshold": 0.46}, MODEL)
    print(f"trained on {len(y)} candidates ({int(y.sum())} windows); saved {MODEL}")
    print("operating point: reject a detector candidate only if P(window) < 0.46")


if __name__ == "__main__":
    main()
