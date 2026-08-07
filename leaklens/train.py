"""
Train the acoustic classifier and the leak-rate regressor.

DELIBERATE CHOICE: random forest / gradient boosting on engineered features,
not a deep CNN.

Reasons, and put these in your report because they are good engineering answers
rather than excuses:

  1. You will have ~150 clips. A CNN overfits that instantly; a forest does not.
  2. It trains in seconds on a laptop, so you can iterate 50 times in an evening.
  3. It runs in microseconds on the QRB2210, which has no NPU.
  4. Feature importances are inspectable, so you can SHOW which acoustic
     properties separate a leak from an impact wrench. A CNN cannot do that
     without extra work, and judges ask.

If you end up with 1000+ clips, features.py already produces log-mel
spectrograms - swap in a small 1-D CNN then. The feature pipeline is shared, so
nothing else changes.

Usage:
    python -m leaklens.train --data data/ --out models/
    python -m leaklens.train --data data/ --out models/ --report
"""

from __future__ import annotations

import argparse
import json
import os
import glob
import warnings

import numpy as np
import joblib

from .features import extract, feature_names, FEATURE_VERSION

CLASSES = ["leak", "tool", "bearing", "background"]


# --------------------------------------------------------------------------
def load_wav(path):
    """Read a WAV without pulling in soundfile/librosa."""
    import wave
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
        width = w.getsampwidth()
        ch = w.getnchannels()
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[width]
    x = np.frombuffer(raw, dtype=dtype).astype(float)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    x /= float(np.iinfo(dtype).max)
    return x, float(sr)


def load_dataset(root, window_s=1.0, hop_s=0.5):
    """
    Expects:
        root/<class_name>/*.wav
    with an optional sidecar  <same name>.json  holding {"flow_lpm": ..., "distance_m": ...}

    Long recordings are chopped into overlapping windows, which multiplies your
    effective dataset size several-fold. With 150 recordings of 10 s each you get
    ~2800 training windows, which is plenty for a forest.
    """
    X, y, meta = [], [], []
    for cls in sorted(os.listdir(root)):
        d = os.path.join(root, cls)
        if not os.path.isdir(d):
            continue
        for path in sorted(glob.glob(os.path.join(d, "*.wav"))):
            try:
                x, sr = load_wav(path)
            except Exception as e:
                warnings.warn(f"skipping {path}: {e}")
                continue

            side = os.path.splitext(path)[0] + ".json"
            info = {}
            if os.path.exists(side):
                with open(side) as f:
                    info = json.load(f)

            win = int(window_s * sr)
            hop = int(hop_s * sr)
            if x.size < win:
                continue
            for start in range(0, x.size - win + 1, hop):
                seg = x[start:start + win]
                if np.sqrt(np.mean(seg ** 2)) < 1e-6:
                    continue                      # digital silence
                X.append(extract(seg, sr))
                y.append(cls)
                meta.append(dict(info, path=path, start=start / sr))
    if not X:
        raise RuntimeError(f"no usable audio found under {root}")
    return np.vstack(X), np.array(y), meta


# --------------------------------------------------------------------------
def train_classifier(X, y, seed=0):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    clf = Pipeline([
        ("scale", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2,
            class_weight="balanced", random_state=seed, n_jobs=-1)),
    ])
    clf.fit(X, y)
    return clf


def evaluate(clf, X, y, groups):
    """
    GROUPED cross-validation, split by source recording.

    This matters enormously. Windows from the same recording are near-duplicates.
    Random splitting leaks them across train and test and reports ~99% accuracy
    that collapses the moment you point the device at a new leak. Group by file
    and you get an honest number to put in the report.
    """
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import classification_report, confusion_matrix

    n_groups = len(set(groups))
    n_splits = min(5, n_groups)
    if n_splits < 2:
        return dict(note="not enough distinct recordings for grouped CV")

    cv = GroupKFold(n_splits=n_splits)
    pred = cross_val_predict(clf, X, y, groups=groups, cv=cv, n_jobs=1)
    labels = sorted(set(y))
    return dict(
        report=classification_report(y, pred, zero_division=0, output_dict=True),
        report_text=classification_report(y, pred, zero_division=0),
        confusion=confusion_matrix(y, pred, labels=labels).tolist(),
        labels=labels,
        n_splits=n_splits,
    )


def train_regressor(X, flows, seed=0):
    """Leak rate in L/min. Trained in log space because flow spans decades."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    mask = np.isfinite(flows) & (flows > 0)
    if mask.sum() < 10:
        return None, dict(note="too few flow-labelled samples; skipped")

    reg = Pipeline([
        ("scale", StandardScaler()),
        ("gb", GradientBoostingRegressor(random_state=seed)),
    ])
    reg.fit(X[mask], np.log(flows[mask]))
    pred = np.exp(reg.predict(X[mask]))
    err = np.abs(pred - flows[mask]) / flows[mask]
    return reg, dict(n=int(mask.sum()),
                     median_rel_error=float(np.median(err)),
                     p90_rel_error=float(np.percentile(err, 90)))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="models")
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--hop", type=float, default=0.5)
    ap.add_argument("--report", action="store_true",
                    help="print feature importances and per-class metrics")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"loading from {args.data} ...")
    X, y, meta = load_dataset(args.data, args.window, args.hop)
    groups = np.array([m["path"] for m in meta])
    print(f"  {X.shape[0]} windows, {X.shape[1]} features, "
          f"{len(set(y))} classes, {len(set(groups))} source recordings")
    for c in sorted(set(y)):
        print(f"    {c:<12} {int((y == c).sum())} windows")

    print("training classifier ...")
    clf = train_classifier(X, y)
    metrics = evaluate(clf, X, y, groups)
    if "report_text" in metrics:
        print(metrics["report_text"])

    flows = np.array([m.get("flow_lpm", np.nan) for m in meta], dtype=float)
    print("training leak-rate regressor ...")
    reg, reg_metrics = train_regressor(X, flows)
    print("  ", reg_metrics)

    bundle = dict(classifier=clf, regressor=reg,
                  feature_version=FEATURE_VERSION,
                  feature_names=feature_names(),
                  classes=sorted(set(y)))
    path = os.path.join(args.out, "leaklens_model.joblib")
    joblib.dump(bundle, path)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(dict(classifier=metrics, regressor=reg_metrics), f, indent=2)
    print(f"saved {path}")

    if args.report:
        rf = clf.named_steps["rf"]
        imp = rf.feature_importances_
        names = feature_names()
        order = np.argsort(imp)[::-1][:15]
        print("\ntop features:")
        for i in order:
            print(f"  {names[i]:<16} {imp[i]:.4f}")
        print("\nPut this table in your report - it shows WHICH acoustic "
              "properties separate the classes.")


if __name__ == "__main__":
    main()
