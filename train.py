#!/usr/bin/env python3
"""
train.py  –  Train / fine-tune the screen detector on your own photos.

Usage:
    python train.py --real real/ --screen screen/

Expects two folders of images.  After training it writes model_weights.npz
which predict.py will auto-load.

Outputs cross-validated accuracy so you have an honest number to report.
"""

import argparse
import os
import glob
import time
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# We import our feature extractor from predict.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from predict import extract_features


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}


def load_folder(folder: str, label: int):
    folder = Path(folder)
    paths  = [p for p in folder.iterdir()
              if p.suffix.lower() in IMAGE_EXTS]
    if not paths:
        raise FileNotFoundError(f"No images found in {folder}")

    features, labels = [], []
    for p in paths:
        try:
            f = extract_features(str(p))
            features.append(f)
            labels.append(label)
        except Exception as e:
            print(f"  [skip] {p.name}: {e}")

    print(f"  Loaded {len(features)} images from {folder}")
    return features, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real",   required=True, help="Folder of real photos")
    parser.add_argument("--screen", required=True, help="Folder of screen photos")
    args = parser.parse_args()

    print("\n── Loading images ──")
    t0 = time.perf_counter()

    real_feats,   real_labels   = load_folder(args.real,   label=0)
    screen_feats, screen_labels = load_folder(args.screen, label=1)

    X = np.array(real_feats + screen_feats, dtype=np.float32)
    y = np.array(real_labels + screen_labels, dtype=np.int32)

    print(f"\n  Total samples: {len(y)}  ({real_labels.count(0)} real, "
          f"{screen_labels.count(1)} screen)")
    print(f"  Feature extraction time: {time.perf_counter() - t0:.1f}s")

    # ── Cross-validated accuracy (honest number) ──
    print("\n── 5-fold cross-validation ──")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced"))
    ])
    n_splits = min(5, np.bincount(y).min())
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    print(f"  Accuracy per fold: {[f'{s:.3f}' for s in scores]}")
    print(f"  Mean ± std:        {scores.mean():.3f} ± {scores.std():.3f}")

    # ── Full-data fit for deployment ──
    print("\n── Training final model on all data ──")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
    clf.fit(X_scaled, y)

    # Training-set report (for reference only)
    y_pred = clf.predict(X_scaled)
    print("\nTraining-set classification report:")
    print(classification_report(y, y_pred, target_names=["real", "screen"]))
    print("Confusion matrix (rows=actual, cols=pred):")
    print(confusion_matrix(y, y_pred))

    # ── Save weights for predict.py ──
    out_path = Path(__file__).parent / "model_weights.npz"
    np.savez(
        out_path,
        weights = clf.coef_[0].astype(np.float32),
        bias    = np.float32(clf.intercept_[0]),
        mu      = scaler.mean_.astype(np.float32),
        sigma   = scaler.scale_.astype(np.float32),
    )
    print(f"\n  Saved model weights → {out_path}")
    print(f"  predict.py will auto-load these weights from now on.")

    print(f"\n── Honest accuracy to report: {scores.mean()*100:.1f}% "
          f"(±{scores.std()*100:.1f}%) ──\n")


if __name__ == "__main__":
    main()