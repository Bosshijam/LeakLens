#!/usr/bin/env python3
"""
LeakLens - end to end scan.

Works TODAY against the mock device, with no hardware whatsoever:

    python run_scan.py --device mock
    python run_scan.py --device mock --leak-az 15 --leak-el -8 --save out/

Against a real Arduino running firmware/leaklens_mcu:

    python run_scan.py --device serial:/dev/ttyUSB0 --model models/leaklens_model.joblib

Against the UNO Q once UnoQBridgeDevice is filled in:

    python run_scan.py --device bridge

The point: this file does not change between those three. Neither does anything
in leaklens/. Only the device implementation swaps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from leaklens.device import open_device, MockDevice
from leaklens import scanner, costing
from leaklens.features import extract


def classify(waveform, sr, model_path):
    if not model_path or not os.path.exists(model_path):
        return None, None
    import joblib
    bundle = joblib.load(model_path)
    feats = extract(waveform, sr).reshape(1, -1)
    clf = bundle["classifier"]
    label = clf.predict(feats)[0]
    proba = None
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(feats)[0]
        proba = dict(zip(clf.classes_, (float(v) for v in p)))
    flow = None
    if bundle.get("regressor") is not None:
        flow = float(np.exp(bundle["regressor"].predict(feats)[0]))
    return (label, proba), flow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--tariff", type=float, default=8.0)
    ap.add_argument("--pressure", type=float, default=7.0)
    ap.add_argument("--save", default=None, help="directory for outputs")
    # mock-only knobs, handy for demos and for testing the scan logic
    ap.add_argument("--leak-az", type=float, default=12.0)
    ap.add_argument("--leak-el", type=float, default=-6.0)
    ap.add_argument("--distance", type=float, default=3.0)
    args = ap.parse_args()

    if args.device == "mock":
        dev = MockDevice(leaks=((args.leak_az, args.leak_el, 1.0),),
                         distance_m=args.distance)
    else:
        dev = open_device(args.device)

    with dev:
        print(dev.ping())
        print("scanning...")

        def progress(frac):
            print(f"\r  {frac*100:5.1f}%", end="", flush=True)

        res = scanner.scan(dev, progress=progress)
        print()

        print(f"  scan took {res.duration_s:.1f} s")
        print(f"  peak at az {res.peak_az:+.2f} deg, el {res.peak_el:+.2f} deg")
        print(f"  SNR {res.snr:.1f} dB  ->  "
              f"{'SOURCE PRESENT' if res.has_source else 'nothing found'}")

        if not res.has_source:
            print("\nNo acoustic source above threshold. Nothing to report.")
            return

        label, flow_from_model = None, None
        if res.waveform is not None:
            out = classify(res.waveform, res.waveform_sr, args.model)
            (label, proba), flow_from_model = out if out[0] else ((None, None), None)
            if label:
                print(f"  classified as: {label}")
                if proba:
                    for k, v in sorted(proba.items(), key=lambda kv: -kv[1]):
                        print(f"      {k:<12} {v:5.1%}")

        if label is not None and label != "leak":
            print(f"\nSource identified as '{label}', not a leak. "
                  f"No cost attributed.")
            dev.set_laser(False)
            return

        cal = None
        if args.calibration and os.path.exists(args.calibration):
            cal = costing.AmplitudeCalibration.load(args.calibration)

        dist = res.distance_m if res.distance_m > 0 else args.distance
        est = costing.estimate(res.peak_value, dist, cal,
                               gauge_bar=args.pressure, tariff=args.tariff)

        print()
        print("=" * 52)
        print(f"  LEAK DETECTED   {est.summary()}")
        print(f"  confidence: {est.confidence}")
        if est.confidence == "uncalibrated":
            print("  !! no calibration loaded - flow figure is a placeholder")
        print("=" * 52)

        dev.set_laser(True)

        if args.save:
            os.makedirs(args.save, exist_ok=True)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.imshow(res.grid, origin="lower", cmap="inferno",
                           extent=[res.az_axis[0], res.az_axis[-1],
                                   res.el_axis[0], res.el_axis[-1]],
                           aspect="auto")
            ax.plot(res.peak_az, res.peak_el, "w+", markersize=16, markeredgewidth=2)
            ax.set_xlabel("azimuth, degrees")
            ax.set_ylabel("elevation, degrees")
            ax.set_title(f"LeakLens  —  {est.summary()}")
            fig.colorbar(im, ax=ax, label="ultrasonic level")
            fig.tight_layout()
            p = os.path.join(args.save, "heatmap.png")
            fig.savefig(p, dpi=130)
            print(f"\nwrote {p}")

            with open(os.path.join(args.save, "result.json"), "w") as f:
                json.dump(dict(
                    peak_az=res.peak_az, peak_el=res.peak_el,
                    snr_db=res.snr, duration_s=res.duration_s,
                    label=label, estimate=json.loads(est.to_json()),
                ), f, indent=2)
            print(f"wrote {os.path.join(args.save, 'result.json')}")


if __name__ == "__main__":
    sys.exit(main())
